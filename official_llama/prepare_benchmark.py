"""Build frozen Llama inputs on the local machine; no GPU/model weights needed.

Official RULER generators are executed unchanged. Never executes their shell
wrapper: arguments are passed as a list, and the exact command is recorded.
"""
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
TASKS = ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa',
         'gov_report', 'qmsum', 'multi_news', 'lcc', 'repobench-p']
METHODS = ['full', 'kivi', 'snapkv', 'scope', 'phasekv']
SEED = 20261029


def sha(data):
    return hashlib.sha256(data).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def used_documents(root):
    used = collections.defaultdict(set)
    ids = collections.defaultdict(set)
    evidence = []
    for path in sorted(root.rglob('*.jsonl')):
        if 'source' in path.parts:
            continue
        found = False
        for line in path.open(encoding='utf-8-sig'):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            match = re.fullmatch(r'quality/([^/]+)/\d+/(\d+)', row.get('case_id', ''))
            if not match:
                match = re.fullmatch(r'([^:]+):(\d+)', str(row.get('sample_id', '')))
            if match and match[1] in TASKS:
                used[match[1]].add(int(match[2]))
                if row.get('dataset_id'):
                    ids[match[1]].add(str(row['dataset_id']))
                found = True
        if found:
            evidence.append({'file': str(path.relative_to(root)), 'sha256': sha(path.read_bytes())})
    return used, ids, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--history', type=Path, default=ROOT.parent / 'results')
    parser.add_argument('--out', type=Path, default=ROOT / 'bench_assets/frozen')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    import yaml
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    eos = json.loads((args.model / 'generation_config.json').read_text())['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    source_hashes = {}

    def add(suite, task, doc, ids, limit, answers=None, **meta):
        entry = dict(id=f'{suite}/{task}/{doc}', suite=suite, task=task,
                     document_id=str(doc), input_ids=ids, max_new_tokens=limit,
                     stop_ids=eos, natural_eos=suite != 'performance',
                     methods=METHODS.copy(), answers=answers, **meta)
        entry['input_sha256'] = sha(json.dumps(ids).encode())
        records.append(entry)
        return entry

    # Exclude historical documents, including other models and truncation lengths.
    used, used_ids, evidence = used_documents(args.history)
    cfgdir = ROOT / 'vendor/SnapKV/experiments/LongBench/config'
    prompts = json.loads((cfgdir / 'dataset2prompt.json').read_text('utf-8'))
    limits = json.loads((cfgdir / 'dataset2maxlen.json').read_text('utf-8'))
    archive = ROOT / 'bench_assets/longbench_data.zip'
    source_hashes['longbench_data.zip'] = sha(archive.read_bytes())
    selection = {}
    split_notes = {}
    with zipfile.ZipFile(archive) as z:
        for task in TASKS:
            data = [json.loads(x) for x in z.read(f'data/{task}.jsonl').decode().splitlines() if x.strip()]
            banned_contexts = {sha(data[i]['context'].encode()) for i in used[task] if i < len(data)}
            pool = [i for i, d in enumerate(data) if i not in used[task]
                    and str(d.get('_id', i)) not in used_ids[task]
                    and sha(d['context'].encode()) not in banned_contexts]
            random.Random(SEED).shuffle(pool)
            chosen, seen_contexts = [], set()
            for i in pool:
                context_hash = sha(data[i]['context'].encode())
                if context_hash not in seen_contexts:
                    chosen.append(i)
                    seen_contexts.add(context_hash)
                if len(chosen) == 32:
                    break
            heldout_indices = set(chosen)
            if len(chosen) != 32:
                # User authorized starting the comparison. Disclose re-evaluation
                # instead of claiming a held-out set that no longer exists.
                fallback = [i for i in range(len(data)) if i not in chosen]
                random.Random(SEED).shuffle(fallback)
                chosen += fallback[:32-len(chosen)]
                print(f'{task}: {len(heldout_indices)} held-out; {32-len(heldout_indices)} historical re-evaluation documents.', flush=True)
            split_notes[task] = {'heldout_documents':len(heldout_indices),
                                'historical_or_shared_context_retest':32-len(heldout_indices)}
            selection[task] = chosen
            for i in chosen:
                row = data[i]
                prompt = prompts[task].format(**row)
                raw = tokenizer.encode(prompt, add_special_tokens=False)
                for budget in (8192, 16384):
                    # Leave room for chat wrappers while preserving head/tail protocol.
                    if task in ('lcc', 'repobench-p'):
                        overhead = 1
                    else:
                        overhead = len(tokenizer.apply_chat_template([{'role':'user','content':''}], tokenize=True, add_generation_prompt=True)) + 8
                    cap = budget - overhead
                    truncated = len(raw) > cap
                    body = tokenizer.decode(raw[:cap//2], skip_special_tokens=True) + tokenizer.decode(raw[-(cap-cap//2):], skip_special_tokens=True) if truncated else prompt
                    if task in ('lcc', 'repobench-p'):
                        ids = tokenizer.encode(body, add_special_tokens=True)
                    else:
                        ids = tokenizer.apply_chat_template([{'role':'user','content':body}], tokenize=True, add_generation_prompt=True)
                    if len(ids) > budget:
                        raise ValueError(f'Prompt boundary overflow: {task} {len(ids)} > {budget}')
                    add('longbench', task, f'{i}/{budget}', ids, limits[task], row['answers'],
                        length_budget=budget, source_index=i, original_id=row.get('_id', str(i)),
                        cluster_id=f'{task}/{sha(row["context"].encode())}', truncated_input=truncated,
                        original_content_tokens=len(raw), heldout_from_available_history=i in heldout_indices,
                        evaluation_split='heldout' if i in heldout_indices else 'historical_retest')
    print('Frozen LongBench: 640 prompts, historical overlap explicitly labeled.', flush=True)

    # Use the author's existing compound prompts and official scoring functions.
    directive = "Answer each question step by step, adhering to the format shown in the examples provided. Start each response with 'Answer_' and introduce the final response with 'The answer is'. Do not repeat the question. Ensure that you respond to all the questions presented, regardless of their number."
    for task, k in [('gsm8k', 30), ('mmlu', 30), ('csqa', 40)]:
        path = ROOT / f'vendor/SCOPE/data/LongGenBench/{task}/{task}_{k}.jsonl'
        source_hashes[str(path.relative_to(ROOT))] = sha(path.read_bytes())
        rows = [json.loads(x) for x in path.read_text('utf-8').splitlines() if x.strip()]
        order = list(range(len(rows)))
        random.Random(SEED).shuffle(order)
        chosen_questions = set()
        picked = 0
        for i in order:
            row = rows[i]
            question_ids = [sha(q.encode()) for q in row['questions']]
            if chosen_questions.intersection(question_ids):
                continue
            chosen_questions.update(question_ids)
            # Faithful official llama3_modify wrapper; no corrected hidden template.
            system = directive + (' The following are multiple choice questions (with answers) about ' if task == 'mmlu' else '')
            prompt = f'<<SYS>>\n {system} \n<</SYS>>\n\n{row["prompt"]}'
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            entry = add('longgenbench', task, str(i), ids, 4096, row['answers'],
                        cluster_id=f'{task}/{i}', question_hashes=question_ids,
                        first_answer_number=9 if task == 'gsm8k' else 6,
                        source_index=i, truncated_input=False, length_budget='4k_output')
            entry['methods'] += ['scope_slide', 'scope_adaptive']
            picked += 1
            if picked == 16:
                break
        if picked != 16:
            raise RuntimeError(f'{task}: fewer than 16 non-overlapping compound prompts')
    print('Frozen LongGenBench: 48 compound prompts, all methods output limit 4096.', flush=True)

    ruler = ROOT / 'bench_sources/RULER/scripts'
    customized = yaml.safe_load((ruler / 'synthetic.yaml').read_text('utf-8'))
    constants = module('ruler_data_constants', ruler / 'data/synthetic/constants.py').TASKS
    template = module('ruler_templates', ruler / 'data/template.py').Templates['meta-llama3']
    env = {**os.environ, 'NLTK_DATA': str(ROOT / 'bench_assets/nltk_data'),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUTF8': '1'}
    commands = []
    for task in ('niah_multikey_1', 'niah_multivalue', 'vt', 'cwe'):
        opts = customized[task]
        base = constants[opts['task']]
        for budget in (4096, 8192, 16384):
            folder = args.out / 'ruler_raw' / str(budget)
            file = folder / task / 'validation.jsonl'
            cmd = [sys.executable, str(ruler / 'data/synthetic' / (opts['task'] + '.py')),
                   '--save_dir', str(folder), '--save_name', task, '--subset', 'validation',
                   '--tokenizer_path', str(args.model.resolve()), '--tokenizer_type', 'hf',
                   '--max_seq_length', str(budget), '--tokens_to_generate', str(base['tokens_to_generate']),
                   '--num_samples', '32', '--random_seed', str(SEED),
                   '--template', template.format(task_template=base['template']) + base.get('answer_prefix', '')]
            for key, value in opts['args'].items():
                cmd += ['--' + key, str(value)]
            commands.append(cmd)
            print(f'Prepare official RULER {task}/{budget}', flush=True)
            if not file.exists():
                subprocess.run(cmd, check=True, env=env)
            rows = [json.loads(x) for x in file.read_text('utf-8').splitlines() if x.strip()]
            if len(rows) != 32:
                raise ValueError('Incomplete RULER generator output')
            source_hashes[str(file.relative_to(args.out))] = sha(file.read_bytes())
            for i, row in enumerate(rows):
                ids = tokenizer.encode(row['input'] + row.get('answer_prefix', ''), add_special_tokens=False)
                if len(ids) + base['tokens_to_generate'] > budget:
                    raise ValueError('RULER total length overflow; do not truncate answer-bearing text')
                add('ruler', task, f'{i}/{budget}', ids, base['tokens_to_generate'], row['outputs'],
                    cluster_id=f'{task}/{i}', length_budget=budget, truncated_input=False,
                    source_index=i, ruler_base_task=opts['task'])

    shapes = [(4096,256),(8192,256),(16384,256),(1024,1024),(1024,4096),
              (1024,8192),(8192,2048),(16384,4096),(32000,256)]
    for length, output in shapes:
        for index, text in enumerate([
            'A library catalog records books about science, history and mathematics. ',
            'The travel journal describes mountains, rivers, forests and ancient cities. ',
            'A software manual explains functions, variables, data structures and examples. ']):
            pattern = tokenizer.encode(text, add_special_tokens=False)
            ids = [tokenizer.bos_token_id] + (pattern * (length//len(pattern)+1))[:length-1]
            add('performance', f'{length}+{output}', str(index), ids, output,
                length_budget=length, capacity_extension=length==32000,
                cluster_id=f'performance/{length}+{output}/{index}')
    target = args.out / 'samples.jsonl.gz'
    with gzip.open(target, 'wt', encoding='utf-8', newline='\n') as f:
        for entry in records:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    manifest = {
        'schema': 1, 'seed': SEED, 'samples_sha256': sha(target.read_bytes()),
        'tokenizer_files': {p.name: sha(p.read_bytes()) for p in args.model.glob('*.json')},
        'counts': dict(collections.Counter(r['suite'] for r in records)),
        'quality_generations': sum(len(r['methods']) for r in records if r['suite']!='performance'),
        'performance_measured': sum(len(r['methods'])*3 for r in records if r['suite']=='performance'),
        'historical_files': evidence, 'excluded_indices': {k:sorted(v) for k,v in used.items()},
        'longbench_selected_indices': selection, 'longbench_split_notes':split_notes, 'data_hashes': source_hashes,
        'ruler_commit': subprocess.check_output(['git','-C',str(ruler.parent),'rev-parse','HEAD'],text=True).strip(),
        'ruler_generation_commands': commands,
        'limits': ['Historical exclusion covers available JSONL evidence; does not prove absence of unrecorded prior use.',
                   'LongGenBench uses upstream compound prompts, not newly reconstructed problems.',
                   'RULER length includes output reserve; LongBench length is maximum input tokens.',
                   'LongGenBench official parser limitations are retained; indexed completion is an additional diagnostic.']}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({k:manifest[k] for k in ('counts','quality_generations','performance_measured')}, indent=2))


if __name__ == '__main__':
    main()
