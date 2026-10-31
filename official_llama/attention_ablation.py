"""Isolated four-way attention/cache ablation; GPU gates precede formal measurements."""
import argparse
import collections
import csv
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ablation_runtime import METHODS, cache_method
SUITES = ['performance', 'longbench', 'ruler', 'longgenbench']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def samples():
    directory = ROOT / 'bench_assets/frozen'
    manifest = json.loads((directory/'manifest.json').read_text('utf-8'))
    path = directory/'samples.jsonl.gz'
    if digest(path.read_bytes()) != manifest['samples_sha256']:
        raise RuntimeError('Frozen input hash mismatch')
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    return rows, manifest


def result_path(out, method, sample, repeat):
    key = digest(sample['id'].encode())[:24]
    return Path(out)/'records'/method/f'{key}_r{repeat}.json'


def expected(rows, suites, methods):
    for sample in rows:
        if sample['suite'] not in suites:
            continue
        for method in methods:
            if cache_method(method) not in sample['methods']:
                continue
            for repeat in range(3 if sample['suite']=='performance' else 1):
                yield method, sample, repeat


def pending(path, retry_failed=False):
    if not path.exists():
        return True
    row = json.loads(path.read_text('utf-8'))
    return retry_failed and row['status'] != 'ok'


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
                             '--format=csv,noheader'], capture_output=True, text=True)
    return {'time': time.time(), 'csv':result.stdout.strip(), 'error':result.stderr.strip()}


def verify_worker_protocol(job):
    out = Path(job['out'])
    manifest = json.loads((out/'run_manifest.json').read_text('utf-8'))
    gate = json.loads((out/'gpu_gate.json').read_text('utf-8'))
    if (manifest['protocol_hash'] != job['protocol_hash'] or gate.get('status') != 'passed'
            or gate.get('protocol_hash') != job['protocol_hash']):
        raise RuntimeError('Worker requires a successful GPU gate for this exact protocol')
    for name, expected_hash in manifest['protocol']['code'].items():
        if digest((ROOT/name).read_bytes()) != expected_hash:
            raise RuntimeError(f'Source changed after the GPU gate: {name}; choose a new output directory')


def worker(job_path):
    import torch
    import flash_attn
    import transformers
    import triton
    from transformers import AutoTokenizer
    from ablation_runtime import load_model, reset_sample
    from ablation_generate import generate
    from benchmark_metrics import score
    from prepare import verify
    verify()
    job = json.loads(Path(job_path).read_text('utf-8'))
    out = Path(job['out'])
    verify_worker_protocol(job)
    method = job['method']
    rows, _ = samples()
    rows = [r for r in rows if r['suite']==job['suite'] and cache_method(method) in r['methods']]
    todo = [(r, rep) for r in rows for rep in range(3 if r['suite']=='performance' else 1)
            if pending(result_path(out,method,r,rep),job['retry_failed'])]
    if not todo:
        return
    config = json.loads((ROOT/'config.json').read_text('utf-8'))
    if flash_attn.__version__ != config['flash_attn']:
        raise RuntimeError('FlashAttention version mismatch')
    torch.manual_seed(20261029)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(job['model'], local_files_only=True)
    print(f'Loading ONE model for {method}/{job["suite"]}; {len(todo)} records pending', flush=True)
    model, options, origin = load_model(job['model'], method)
    atomic(out/'environment'/f'{job["suite"]}__{method}.json', {
        'torch':torch.__version__, 'transformers':transformers.__version__,
        'flash_attn':flash_attn.__version__, 'triton':triton.__version__,
        'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(),
        'dtype':'float16', 'batch_size':1, 'options':options, 'source':origin,
        'cache_policy':cache_method(method),
        'decode_backend':'flash_attention_2' if method.endswith('_fa2') else 'common_segmented_triton',
        'materializes_fp16_kv_each_layer_step':method=='phasekv_fa2',
        'attention_classes':sorted({type(x.self_attn).__name__ for x in model.model.layers}),
        'device':str(next(model.parameters()).device), 'pid':os.getpid()})
    warmed = set()
    for number, (sample, repeat) in enumerate(todo, 1):
        print(f'CASE {number}/{len(todo)} {method} {sample["id"]} repeat={repeat}', flush=True)
        atomic(out/'progress.json', {'method':method,'suite':job['suite'],'sample':sample['id'],
               'repeat':repeat,'worker_done':number-1,'worker_total':len(todo),'time':time.time()})
        base = {k:sample.get(k) for k in ('id','suite','task','document_id','cluster_id','length_budget',
                                         'input_sha256','truncated_input','capacity_extension',
                                         'evaluation_split','heldout_from_available_history')}
        base.update(method=method, repeat=repeat, dtype='float16', protocol_hash=job['protocol_hash'])
        diagnostic = out/'diagnostics'/method/(digest(sample['id'].encode())[:24]+'.json')
        stage = 'generation'
        result = None
        try:
            if sample['suite']=='performance' and sample['id'] not in warmed:
                stage = 'warmup_storage_diagnostic'
                # This also warms every tested shape; its timings NEVER enter speed tables.
                warm = generate(model, method, options, sample, diagnostic=True)
                warm.pop('output_ids')
                atomic(diagnostic, {**base, **warm, 'timing_excluded':True})
                del warm
                warmed.add(sample['id'])
            stage = 'generation'
            gc.collect()
            torch.cuda.empty_cache()
            before = gpu_snapshot()
            result = generate(model, method, options, sample)
            prediction = tokenizer.decode(result['output_ids'], skip_special_tokens=True)
            stage = 'scoring'
            metrics = score(sample, prediction)
            row = {**base, **result, **metrics, 'status':'ok', 'prediction':prediction,
                   'gpu_before':before, 'gpu_after':gpu_snapshot(), 'finished_at':time.time()}
        except torch.OutOfMemoryError as error:
            row = {**base,'status':'oom','stage':stage,'error':str(error),
                   'traceback':traceback.format_exc(), 'finished_at':time.time()}
        except Exception as error:
            # Unexpected errors cannot silently poison thousands of later samples.
            row = {**base,'status':'error','stage':stage,'error':repr(error),
                   'traceback':traceback.format_exc(), 'finished_at':time.time()}
            if result is not None:
                row['generated_result_before_scoring_error'] = result
            atomic(result_path(out,method,sample,repeat),row)
            raise
        atomic(result_path(out,method,sample,repeat),row)
        print(f'RESULT {row["status"]}: {sample["id"]} score={row.get("score")} '
              f'decode={row.get("decode_tokens_per_s")} token/s', flush=True)
        # Exception traceback must not retain failed forward tensors across samples.
        result = None
        del row
        reset_sample(model,method,options,sample['max_new_tokens'])
        gc.collect()
        torch.cuda.empty_cache()


def summarize(out, rows, suites, methods):
    out = Path(out)
    groups = collections.defaultdict(list)
    all_rows = []
    status = collections.Counter()
    for method,sample,repeat in expected(rows,suites,methods):
        path = result_path(out,method,sample,repeat)
        row = json.loads(path.read_text('utf-8')) if path.exists() else {
            'method':method,'id':sample['id'],'suite':sample['suite'],'task':sample['task'],
            'length_budget':sample.get('length_budget'),'repeat':repeat,'status':'pending'}
        status[row['status']] += 1
        all_rows.append(row)
        groups[(sample['suite'],sample['task'],str(sample.get('length_budget')),method,
                sample.get('evaluation_split','new_frozen'))].append(row)
    compact = []
    for (suite,task,budget,method,split), entries in groups.items():
        ok = [r for r in entries if r['status']=='ok']
        scored = [r['score'] for r in ok if 'score' in r]
        speeds = [r['decode_tokens_per_s'] for r in ok if r.get('decode_tokens_per_s') is not None]
        compact.append({'suite':suite,'task':task,'budget':budget,'method':method,'evaluation_split':split,
                        'expected':len(entries),'ok':len(ok),
                        'oom':sum(r['status']=='oom' for r in entries),
                        'error':sum(r['status']=='error' for r in entries),
                        'pending':sum(r['status']=='pending' for r in entries),
                        'score_success_only':statistics.mean(scored) if scored else None,
                        'score_system_failures_zero':sum(scored)/len(entries) if suite!='performance' and not any(r['status']=='pending' for r in entries) else None,
                        'decode_tokens_per_s_median_descriptive':statistics.median(speeds) if speeds else None,
                        'final_kv_mib_median':statistics.median(r['final_storage']['physical_bytes']/2**20 for r in ok) if ok else None,
                        'peak_allocated_gib_max':max(r['peak_allocated_bytes']/2**30 for r in ok) if ok else None})
    atomic(out/'summary.json', {'status':dict(status),'groups':compact,
           'completed':status['pending']==0,'all_succeeded':status['pending']==0 and not status['oom'] and not status['error'],
           'notes':['Quality metrics differ by task; do not call their pooled mean accuracy.',
                    'Timing on natural quality outputs is descriptive, not the fixed-length speed comparison.',
                    'Persistent KV peaks come from separate warmup diagnostics; allocator peaks include temporary memory.',
                    'NVML snapshots are sampled endpoints, not peak process memory.']})
    if compact:
        with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(compact[0])); writer.writeheader(); writer.writerows(compact)
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for row in all_rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    perf = []
    for sample in rows:
        if sample['suite']!='performance':
            continue
        for method in methods:
            entries = [r for r in all_rows if r['method']==method and r['id']==sample['id'] and r['status']=='ok']
            if not entries:
                continue
            values = [r['decode_tokens_per_s'] for r in entries]
            perf.append({'method':method,'shape':sample['task'],'prompt_id':sample['document_id'],
                         'successful_repeats':len(values),'expected_repeats':3,
                         'decode_median':statistics.median(values),'decode_min':min(values),'decode_max':max(values),
                         'ttft_median_s':statistics.median(r['ttft_s'] for r in entries)})
    atomic(out/'performance_by_prompt.json',perf)
    # Document-cluster paired bootstrap, never token-level resampling.
    import numpy as np
    rng=np.random.default_rng(20261029)
    full={r['id']:r for r in all_rows if r['method']=='full_fa2' and r['status']=='ok' and 'score' in r}
    paired=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in all_rows:
        if r['method']=='full_fa2' or r['status']!='ok' or 'score' not in r or r['id'] not in full:
            continue
        baseline=full[r['id']]
        if r['input_sha256']!=baseline['input_sha256']:
            raise RuntimeError('Input pairing mismatch')
        key=(r['suite'],r['task'],r['method'],r.get('evaluation_split') or 'new_frozen')
        paired[key][r['cluster_id']].append(r['score']-baseline['score'])
    intervals=[]
    for key,clusters in paired.items():
        values=np.array([statistics.mean(v) for v in clusters.values()])
        ci=np.percentile(rng.choice(values,(2000,len(values))).mean(axis=1),[2.5,97.5]).tolist() if len(values)>1 else [None,None]
        intervals.append({'suite':key[0],'task':key[1],'method':key[2],'evaluation_split':key[3],
                          'paired_document_clusters':len(values),'delta_vs_full_points':float(values.mean()),
                          'bootstrap95_low':ci[0],'bootstrap95_high':ci[1],
                          'scope':'successful pairs only; failures remain in summary.csv'})
    atomic(out/'quality_paired_vs_full.json',intervals)
    text = '# Llama attention/cache ablation results\n\n'+json.dumps(dict(status),ensure_ascii=False)+'\n\n'
    text += 'See summary.csv for per-task quality, failures and storage; performance_by_prompt.json for repeated fixed-length speed.\n'
    text += 'Failed and pending samples are retained. This is not an all-success statement.\n'
    (out/'summary.md').write_text(text,encoding='utf-8')
    from ablation_analysis import paired_report
    paired_report(out, all_rows)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='/data/Llama-3.1-8B-Instruct/snapshots/master')
    parser.add_argument('--out',default='/data/phasekv-results/attention_ablation_20261031')
    parser.add_argument('--suites',nargs='+',choices=SUITES,default=SUITES)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--prepare-only',action='store_true',help='Validate frozen data/model identity without loading GPU weights')
    parser.add_argument('--gate-only',action='store_true',help='Run GPU correctness and real-model smoke checks only')
    args=parser.parse_args()
    if len(args.methods) != len(set(args.methods)) or len(args.suites) != len(set(args.suites)):
        parser.error('Duplicate methods/suites are not valid independent measurements')
    rows, data_manifest=samples()
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        manifest=json.loads((out/'run_manifest.json').read_text('utf-8'))
        protocol=manifest['protocol']
        summarize(out,rows,protocol['suites'],protocol['methods']); return 0
    from prepare import verify
    verify()
    from benchmark_metrics import score
    for suite in ('longbench','ruler','longgenbench'):
        score(next(s for s in rows if s['suite']==suite),'')
    for name,expected_hash in data_manifest['tokenizer_files'].items():
        if name=='configuration.json':
            continue  # ModelScope metadata is not used by Transformers.
        path=Path(args.model)/name
        if not path.exists() or digest(path.read_bytes())!=expected_hash:
            raise RuntimeError(f'Model/tokenizer JSON differs from frozen inputs: {name}')
    source_files=['attention_ablation.py','ablation_runtime.py','ablation_attention.py',
                  'ablation_kernels.py','ablation_generate.py','ablation_validate.py','ablation_analysis.py',
                  'benchmark.py','benchmark_generate.py','benchmark_metrics.py','native_runtime.py','prepare.py','config.json','sources.json',
                  'bench_sources/RULER/scripts/eval/synthetic/constants.py',
                  'vendor/SnapKV/experiments/LongBench/metrics.py','vendor/SCOPE/eval_longgenbench.py']
    protocol={'frozen_data':data_manifest['samples_sha256'],
              'code':{p:digest((ROOT/p).read_bytes()) for p in source_files},
              'model':str(Path(args.model).resolve()),'suites':args.suites,'methods':args.methods,'experiment':'attention-cache-2x2-v1',
              'weight_files':{p.name:p.stat().st_size for p in sorted(Path(args.model).glob('*.safetensors'))}}
    protocol_hash=digest(json.dumps(protocol,sort_keys=True).encode())
    manifest_file=out/'run_manifest.json'
    if manifest_file.exists():
        old=json.loads(manifest_file.read_text('utf-8'))
        if old['protocol_hash']!=protocol_hash:
            raise RuntimeError('Resume protocol changed. Choose a new --out; never mix batches.')
    else:
        atomic(manifest_file,{'protocol_hash':protocol_hash,'protocol':protocol,'frozen_manifest':data_manifest})
    total=sum(1 for _ in expected(rows,args.suites,args.methods))
    print(f'ATTENTION ABLATION: {total} measured records; resumable output={out}',flush=True)
    if args.prepare_only:
        print('Frozen inputs and model identity verified; no GPU test executed.'); return 0
    # Lock in Python too, so direct invocation cannot bypass the shell launch lock.
    import fcntl
    lock_path=Path(os.environ.get('PHASEKV_BENCHMARK_LOCK','/data/phasekv-native-benchmark.lock'))
    lock=lock_path.open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('A formal benchmark is already running; refusing duplicate.',flush=True); return 75
    live=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if live.stdout.strip():
        raise RuntimeError('GPU has existing compute processes; refusing to compete:\n'+live.stdout)
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (out/'pip-freeze.txt').write_text(freeze.stdout,encoding='utf-8')
    atomic(out/'gpu_initial.json',gpu_snapshot())
    child=None
    def interrupted(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
    try:
        gate_path=out/'gpu_gate.json'
        atomic(gate_path, {'status':'running','protocol_hash':protocol_hash})
        gate_log=out/'gpu_gate.log'
        with gate_log.open('a',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'ablation_validate.py'),
                                    '--out',str(gate_path),'--model',args.model,
                                    '--protocol-hash',protocol_hash],
                                   stdout=log,stderr=subprocess.STDOUT,env=env)
            print(f'GPU correctness + real-model smoke gate: {gate_log}',flush=True)
            code=child.wait()
        gate=json.loads(gate_path.read_text('utf-8'))
        if code or gate.get('status')!='passed' or gate.get('protocol_hash')!=protocol_hash:
            raise RuntimeError(f'GPU gate failed; no formal measurements started. See {gate_log}')
        if args.gate_only:
            print('GPU gate passed; formal measurements not requested.',flush=True)
            return 0
        for suite_index,suite in enumerate(args.suites):
            order=args.methods[suite_index:]+args.methods[:suite_index]
            for method in order:
                todo=[(s,r) for m,s,r in expected(rows,[suite],[method]) if pending(result_path(out,m,s,r),args.retry_failed)]
                if not todo:
                    continue
                job=out/'jobs'/f'{suite}__{method}.json'
                atomic(job,{'out':str(out),'model':args.model,'suite':suite,'method':method,
                            'retry_failed':args.retry_failed,'protocol_hash':protocol_hash})
                log_path=out/'logs'/f'{suite}__{method}.log'; log_path.parent.mkdir(exist_ok=True)
                print(f'=== FORMAL {suite}/{method}: {len(todo)} pending; log={log_path} ===',flush=True)
                with log_path.open('a',encoding='utf-8') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'__worker__',str(job)],
                                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                           text=True,encoding='utf-8',errors='replace',env=env)
                    for line in child.stdout:
                        log.write(line); log.flush(); print(line,end='',flush=True)
                    code=child.wait()
                summarize(out,rows,args.suites,args.methods)
                if code:
                    raise RuntimeError(f'{suite}/{method} worker failed ({code}); other work remains pending. See {log_path}')
    finally:
        status=summarize(out,rows,args.suites,args.methods)
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
    print('ATTENTION ABLATION FINISHED: '+json.dumps(dict(status)),flush=True)
    print(f'Results: {out}/summary.csv',flush=True)
    return 2 if status['oom'] or status['error'] or status['pending'] else 0


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='__worker__':
        worker(sys.argv[2])
    else:
        raise SystemExit(main())
