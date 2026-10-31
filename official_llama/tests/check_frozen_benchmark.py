"""Audit actual frozen inputs, official prefixes, counts and split labels."""
from pathlib import Path
import collections
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import benchmark


def main():
    from transformers import AutoTokenizer
    model=Path(sys.argv[1])
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    vocab_size=len(tokenizer)
    rows,manifest=benchmark.samples()
    counts=collections.Counter(r['suite'] for r in rows)
    assert dict(counts)=={'longbench':640,'longgenbench':48,'ruler':384,'performance':27}, counts
    assert len(list(benchmark.expected(rows,benchmark.SUITES,benchmark.METHODS+['scope_slide','scope_adaptive'])))==5861
    lookup={r['id']:r for r in rows}
    for sample in rows:
        assert sample['input_sha256']==benchmark.digest(json.dumps(sample['input_ids']).encode())
        assert len(sample['input_ids'])+sample['max_new_tokens']<=131072
        assert sample['input_ids'] and all(isinstance(t,int) and 0<=t<vocab_size for t in sample['input_ids'])
        assert sample['natural_eos']==(sample['suite']!='performance')
        assert len(sample['stop_ids'])>0
        if sample['suite']=='performance':
            assert len(sample['input_ids'])==sample['length_budget']
        if sample['suite']=='longbench':
            assert len(sample['input_ids'])<=sample['length_budget']
            assert sample['evaluation_split'] in ('heldout','historical_retest')
    for path in (ROOT/'bench_assets/frozen/ruler_raw').glob('*/*/validation.jsonl'):
        budget=int(path.parent.parent.name); task=path.parent.name
        raw=[json.loads(l) for l in path.read_text('utf-8').splitlines()]
        for i,entry in enumerate(raw):
            sample=lookup[f'ruler/{task}/{i}/{budget}']
            assert sample['answers']==entry['outputs']
            assert sample['input_ids']==tokenizer.encode(entry['input']+entry.get('answer_prefix',''),add_special_tokens=False)
    for task in ('gsm8k','mmlu','csqa'):
        seen=set()
        for sample in rows:
            if sample['suite']=='longgenbench' and sample['task']==task:
                assert not seen.intersection(sample['question_hashes'])
                seen.update(sample['question_hashes'])
                assert len(sample['answers'])==(40 if task=='csqa' else 30)
    output={'status':'passed','counts':dict(counts),'quality_generations':5456,
            'performance_measured':405,'frozen_sha256':manifest['samples_sha256'],
            'ruler_prefix_roundtrip':'all 384 exact','model_context_bounds':'all passed',
            'longgenbench_question_overlap':'none within selected task',
            'longbench_splits':manifest['longbench_split_notes'],
            'gpu_execution':'not run by this check'}
    (ROOT/'validation/benchmark_frozen_checks.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
