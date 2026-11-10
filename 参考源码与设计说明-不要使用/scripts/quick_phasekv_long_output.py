"""Long-output model throughput: saved PhaseKV baseline versus opt-in P5 only."""
import argparse
from datetime import datetime
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_phasekv_optimizations as opt

b, t, support = opt.b, opt.t, opt.support
HERE = Path(__file__).resolve()
VERSION = 'phasekv-long-output-quick-v1'


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--out', type=Path, default=ROOT/'results'/('phasekv_long_output_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--prompts', nargs='+', type=int, default=[1024])
    p.add_argument('--tokens', type=int, default=4096)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--report-only', action='store_true')
    a = p.parse_args(argv)
    if a.repeats < 1 or a.tokens < 1024 or min(a.prompts) < 32 or len(set(a.prompts)) != len(a.prompts):
        p.error('Require repeats>=1, output>=1024, unique prompts>=32')
    a.out = a.out.resolve()
    return a


def plan(a):
    base = b.parse_args(['--models','qwen','--methods','phasekv','--qwen',a.model_path,
                         '--out',str(a.out),'--repeats',str(a.repeats),'--seed',str(a.seed)])
    profile = b.profiles_for(base)[0]
    limit = min(base.max_context, profile['architecture']['max_position_embeddings'])
    if any(n+a.tokens > limit or n > limit-128 for n in a.prompts):
        raise ValueError('Requested prompt + output exceeds the model context limit')
    settings = dict(vars(base), phases=['performance'], tasks=[],
                    performance_shapes=[(n,a.tokens,'long_output_quick') for n in a.prompts])
    all_specs = {name: spec for name,spec,_ in opt.configs()}
    profile['methods'] = [all_specs['PhaseKV-baseline'],all_specs['PhaseKV']]
    return dict(version=VERSION, profile=profile, settings=settings,
                groups=[('baseline',all_specs['PhaseKV-baseline']),('p5',all_specs['PhaseKV'])],
                generations_per_group=len(a.prompts)*(a.repeats+1),
                note='Same matched prefill; synthetic inputs; one full-length warmup per shape excluded; no LongBench/Full/other algorithms.')


def gpu_guard(output):
    """Targeted guard, not a substitute for the full delivery GPU test suite."""
    if os.environ.get('TRITON_INTERPRET') == '1':
        raise RuntimeError('Actual CUDA compilation is required; unset TRITON_INTERPRET')
    b.check_environment()
    support.bootstrap()
    import torch
    from scripts.bench_phasekv_optimizations import inputs_for, cache_factory, assert_storage_equal
    config = support.load_config_module().stage_configs()['p5'].to_dict()
    support.load_baseline()
    old = importlib.import_module('phasekv_saved_baseline.kernels.segmented_attention').segmented_phasekv_decode
    new = importlib.import_module('PhaseKV.kernels.segmented_attention').segmented_phasekv_decode
    checks = []
    with torch.inference_mode():
        # Cover grouping threshold, page growth, long-output cold count, hot wrap.
        for count in (0,7,31,247):
            inputs = inputs_for(1024,count,143,28,4,1234+count,'cuda')
            reference = cache_factory(inputs,{},True)
            actual = cache_factory(inputs,config)
            q = inputs[5]
            token = inputs[0][:,:,:1].to(torch.float16)
            worst = 0.
            for step in range(18):
                assert len(reference.layers[0].cold) == len(actual.layers[0].cold)
                assert_storage_equal(reference,actual)
                expected = old(q,reference.layers[0],28)
                result = new(q,actual.layers[0],28)
                torch.testing.assert_close(result,expected,atol=.002,rtol=.01,equal_nan=False)
                worst = max(worst,float((result.float()-expected.float()).abs().max().item()))
                if step < 17:
                    for cache in (reference,actual):
                        cache.append_hot(0,token,-token)
                    # Production attends before cooling this token, including hot_len=144.
                    expected = old(q,reference.layers[0],28)
                    result = new(q,actual.layers[0],28)
                    torch.testing.assert_close(result,expected,atol=.002,rtol=.01,equal_nan=False)
                    for cache in (reference,actual):
                        cache.finish_decode_step()
            checks.append(dict(initial_cold=count,steps=18,max_abs_error=worst))
            print(f'[GPU guard] cold={count}: passed',flush=True)
            del reference,actual,inputs
    b.atomic_json(output,dict(status='passed',checks=checks,gpu=torch.cuda.get_device_name(),
                             scope='target Q28/KV4/HD128 P5; packed bytes + attention + wrap/growth; not full quality validation'))


def load_rows(out, manifest, stage):
    path = out/stage/'records.jsonl'
    job = t.read(out/stage/'job.json')
    spec = dict(manifest['groups'])[stage]
    if job['spec'] != spec or job['protocol_hash'] != b.digest({'manifest':manifest['hash'],'stage':stage}):
        raise ValueError('Job fingerprint mismatch: '+stage)
    rows = b.read_records(path)
    expected = {f'performance/long_output_quick/{n}/{m}/{i}'
                for n,m,_ in manifest['settings']['performance_shapes']
                for i in range(manifest['settings']['repeats']+1)}
    if len(rows)!=len(expected) or {r['case_id'] for r in rows}!=expected:
        raise ValueError('Missing or duplicate records: '+stage)
    for r in rows:
        if r['status']!='ok':
            raise ValueError(f"Failed case, no speed conclusion: {stage}/{r['case_id']} {r['status']}")
        if r['protocol_hash']!=job['protocol_hash'] or r['parameters']!=spec or r.get('prefill_policy')!='matched':
            raise ValueError('Record fingerprint mismatch')
        if not r.get('fixed_output') or r['output_tokens']!=r['requested_output_tokens']:
            raise ValueError('Output stopped early')
        if r.get('decode_compression_steps',0)<1:
            raise ValueError('No decode compression observed')
        for key in ('decode_tokens_per_s','decode_s','total_s','final_kv_bytes','peak_allocated_bytes'):
            if not math.isfinite(r[key]) or r[key]<=0:
                raise ValueError('Invalid measurement: '+key)
    return rows


def summarize(out):
    manifest = t.read(out/'manifest.json')
    if t.read(out/'gpu_guard.json').get('status')!='passed':
        raise ValueError('GPU guard did not pass')
    rows = {stage:load_rows(out,manifest,stage) for stage,_ in manifest['groups']}
    baseline = {r['case_id']:r for r in rows['baseline']}
    for r in rows['p5']:
        t.verify_pair(r['case_id'],r,baseline.get(r['case_id']))
    result = dict(version=VERSION,manifest_hash=manifest['hash'],
        record_hashes={k:b.file_hash(out/k/'records.jsonl') for k in rows},shapes=[],
        notes=['Decode includes cache maintenance; excludes prefill. Warmups excluded.',
               'Baseline then P5 sequential blocks; possible clock/temperature order effects.',
               'Synthetic fixed-length generation, not an accuracy or long-writing quality evaluation.'])
    lines = ['PhaseKV 长输出快速验证：保存的优化前版本 vs P5','']
    for n,m,_ in manifest['settings']['performance_shapes']:
        shape = dict(input_tokens=n,output_tokens=m,stages={})
        for stage in rows:
            rr = [r for r in rows[stage] if r['input_tokens']==n and r['output_tokens']==m and not r['is_warmup']]
            if len(rr)!=manifest['settings']['repeats']:
                raise ValueError('Missing measured repeats')
            speeds = [r['decode_tokens_per_s'] for r in rr]
            shape['stages'][stage] = dict(repeats=len(rr),decode_tokens_per_s=statistics.median(speeds),
                samples_tokens_per_s=speeds,decode_s=statistics.median(r['decode_s'] for r in rr),
                prefill_and_compress_s=statistics.median(r['prefill_and_compress_s'] for r in rr),
                total_s=statistics.median(r['total_s'] for r in rr),
                kv_mib=statistics.median(r['final_kv_bytes'] for r in rr)/2**20,
                peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rr)/2**30)
        a,z = shape['stages']['baseline'],shape['stages']['p5']
        speedup = z['decode_tokens_per_s']/a['decode_tokens_per_s']
        shape.update(decode_speedup=speedup,throughput_change_percent=100*(speedup-1),
                     total_speedup=a['total_s']/z['total_s'],
                     timing_ranges_overlap=not(max(a['samples_tokens_per_s'])<min(z['samples_tokens_per_s']) or
                                              max(z['samples_tokens_per_s'])<min(a['samples_tokens_per_s'])))
        result['shapes'].append(shape)
        lines += [f'输入 {n}，输出 {m}，正式重复 {a["repeats"]} 次',
                  f'优化前：{a["decode_tokens_per_s"]:.2f} token/s；P5：{z["decode_tokens_per_s"]:.2f} token/s',
                  f'解码吞吐变化：{100*(speedup-1):+.2f}%（{speedup:.3f}x）；总耗时加速比：{shape["total_speedup"]:.3f}x',
                  f'实际KV：{a["kv_mib"]:.2f} -> {z["kv_mib"]:.2f} MiB',
                  f'原始吞吐：{a["samples_tokens_per_s"]} -> {z["samples_tokens_per_s"]}',
                  '单次初筛，不能判断稳定收益。' if a['repeats']==1 else
                  ('两组测量范围重叠，需增加重复确认。' if shape['timing_ranges_overlap'] else '测量范围不重叠；仍非统计显著性检验。'),'']
    lines += ['本轮不测试准确率；未测Full/其他算法；生成内容可能分叉，但输入、首token和输出长度一致。',
              '预热不计入结果。基线先跑、P5后跑；小幅差异需在相同GPU负载下复测。']
    b.atomic_json(out/'summary.json',result)
    (out/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)
    return result


def main(argv=None):
    a = arguments(argv)
    if a.report_only:
        summarize(a.out);return
    p = plan(a)
    if a.dry_run:
        print(json.dumps(p,ensure_ascii=False,indent=2));return
    a.out.mkdir(parents=True,exist_ok=False)  # Never append to another experiment.
    opt.configure()
    s = opt.same.s
    print(f'Results: {a.out}',flush=True)
    s.execute([sys.executable,'-u',str(HERE),'__guard__',str(a.out/'gpu_guard.json')],a.out/'gpu_guard.log')
    if t.read(a.out/'gpu_guard.json')['status']!='passed':
        raise RuntimeError('GPU guard failed')
    b.atomic_json(a.out/'identity_job.json',[p['profile']])
    s.execute([sys.executable,str(ROOT/'scripts/benchmark_4090d.py'),'__identity__',
               str(a.out/'identity_job.json'),str(a.out/'identity.json')],a.out/'identity.log')
    p['checkpoint'] = t.read(a.out/'identity.json')
    p['source_hashes'] = b.source_hashes()
    for path in (HERE,ROOT/'run_phasekv_long_output.sh'):
        p['source_hashes'][path.relative_to(ROOT).as_posix()] = b.file_hash(path)
    p['hash'] = b.digest(p)
    b.atomic_json(a.out/'manifest.json',p)
    for index,(stage,spec) in enumerate(p['groups'],1):
        print(f'\n=== long-output: {index}/2 {stage} ===',flush=True)
        folder = a.out/stage;folder.mkdir()
        job = dict(name=p['profile']['name'],architecture=p['profile']['architecture'],spec=spec,
            args=p['settings'],prefill_policy='matched',
            model_path=p['checkpoint'][p['profile']['name']]['resolved_path'],records_path=str(folder/'records.jsonl'),
            protocol_hash=b.digest({'manifest':p['hash'],'stage':stage}),
            matched_baseline=str(a.out/'baseline/records.jsonl') if stage=='p5' else None)
        b.atomic_json(folder/'job.json',job)
        s.execute([sys.executable,'-u',str(HERE),'__worker__',str(folder/'job.json')],folder/'run.log')
        load_rows(a.out,p,stage)
    summarize(a.out)
    b.atomic_json(a.out/'complete.json',{'status':'complete','manifest_hash':p['hash']})
    print(f'Complete: {a.out/"summary.txt"}',flush=True)


if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1]=='__worker__':
            opt.configure()
            job = t.read(sys.argv[2])
            with t.worker_policy('matched',job.get('matched_baseline')):
                b.worker(job)
        elif len(sys.argv)>1 and sys.argv[1]=='__guard__':
            gpu_guard(sys.argv[2])
        else:
            main()
    except KeyboardInterrupt:
        print('Interrupted. Evidence retained; use a fresh --out for a new run.',file=sys.stderr)
        sys.exit(130)
