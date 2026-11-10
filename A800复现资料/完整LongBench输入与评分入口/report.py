from pathlib import Path
from collections import Counter
import argparse
import hashlib
import statistics
from common import METHODS,read,rows,result,write,now,speed_cases,speed_result

def summary(out):
    out=Path(out);config=read(out/'config.json',{});state=read(out/'recovery_queue.json',{})
    data=dict(time=now(),policy='320 original full documents, B1 quality, no truncation; speed B1/2/4/8 only',methods={},
              stage=state.get('phase','prepared'),jobs=state.get('jobs',{}),speed_batches=[1,2,4,8])
    lines=['A800 FULL-TEXT RECOVERY','Quality: 320 full LongBench + 112 unchanged other cases/method, B1. Pilot: 10/method.',
           'Speed: 8192/32768 input x 256/2048 output, B1/2/4/8 (ClusterKV/MILLION B1).',
           'method       pilot    all_ok  reused  oom unsupported error pending  mean paired_delta']
    all_rows=rows();full={r['id']:result(out,'full',r) for r in all_rows}
    for method in METHODS:
        records=[result(out,method,r) for r in all_rows]
        counts=Counter(r.get('status','pending') for r in records)
        pilot=[x for x,r in zip(records,all_rows) if r['pilot']]
        good=[r for r in records if r.get('status')=='ok']
        pair=[(r,full[r['id']]) for r in good if full.get(r['id'],{}).get('status')=='ok']
        detail=dict(expected=320,**{k:counts[k] for k in ('ok','oom','unsupported','error','pending')},
            reused=sum(bool(r.get('reused_from')) for r in records),
            pilot_terminal=sum(r.get('status') in ('ok','oom','unsupported') for r in pilot),
            mean_score=statistics.mean(r['score'] for r in good) if good else None,
            paired_n=len(pair),paired_delta=statistics.mean(r['score']-f['score'] for r,f in pair) if pair else None,
            pilot_mean=statistics.mean(r['score'] for r in pilot if r.get('status')=='ok') if any(r.get('status')=='ok' for r in pilot) else None,
            last_error=read(out/'results'/method/'last_error.json'))
        data['methods'][method]=detail
        fmt=lambda x:'-' if x is None else f'{x:.2f}'
        lines.append(f"{method:12} {detail['pilot_terminal']:2}/10 {detail['ok']:8} {detail['reused']:7} {detail['oom']:4} {detail['unsupported']:11} {detail['error']:5} {detail['pending']:7} {fmt(detail['mean_score']):>7} {fmt(detail['paired_delta']):>12}")
        if detail['last_error']:lines.append('  '+detail['last_error'].get('message','worker failed')[:220])
    if state:
        c=Counter(j['status'] for j in state.get('jobs',{}).values());lines.append('Queue jobs: '+str(dict(c)))
        for gpu,info in state.get('lanes',{}).items():lines.append('GPU '+gpu+': '+str(info))
    data['speed_points_expected']=104;data['speed']={};data['other_quality']={}
    if config:
        old=Path(config['old_out']);matrix=read(Path(config['kit'])/'matrix.json')
        other=[r for r in matrix['quality'] if not r['group'].startswith('longbench/')]
        protocols={m:read(old/'results'/m/'protocol.json',{}) for m in METHODS}
        full_gpu=protocols['full'].get('gpu',{}).get('uuid')
        for method in METHODS:
            points=[]
            for case,b in speed_cases(config['bench'],method):
                r=speed_result(old,method,case,b);status=r.get('status','pending')
                point=dict(shape=case['shape'],batch=b,status=status)
                if status=='ok':
                    med=lambda field:statistics.median(x[field] for x in r['runs'])
                    point.update(aggregate_tokens_per_s=med('aggregate_tokens_per_s'),ttft_s=med('ttft_s'),
                        per_sequence_tokens_per_s=med('per_sequence_tokens_per_s'),peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in r['runs']))
                    f=speed_result(old,'full',case,b)
                    if f.get('status')=='ok':point['relative_to_full']=point['aggregate_tokens_per_s']/statistics.median(x['aggregate_tokens_per_s'] for x in f['runs'])
                points.append(point)
            counts=Counter(r['status'] for r in points)
            gpu=protocols[method].get('gpu',{}).get('uuid')
            data['speed'][method]=dict(expected=len(points),counts=dict(counts),points=points,gpu_uuid=gpu,
                same_physical_gpu_as_full=bool(gpu and full_gpu and gpu==full_gpu),concurrent_four_gpu=True)
            records=[]
            for case in other:
                p=old/'results'/method/'quality'/(hashlib.sha256(case['id'].encode()).hexdigest()+'.json')
                r=read(p,{})
                if r and r.get('input_sha256')!=case['input_sha256']:raise ValueError('Other quality input differs: '+str(p))
                records.append(dict(r,group=case['group']))
            counts_other=Counter(r.get('status','pending') for r in records)
            data['other_quality'][method]=dict(expected=len(other),counts=dict(counts_other),source=str(old/'results'/method/'quality'))
            lines.append(f"{method:12} speed {counts['ok']+counts['oom']:3}/{len(points)} (ok={counts['ok']} oom={counts['oom']}); other quality {counts_other['ok']+counts_other['oom']}/{len(other)}")
    lines.append('Speed target: 104 points. Old real-workload/B3/5/6/7 results stay archived, outside this matrix.')
    lines.append('Speed ratios across different UUIDs are concurrent four-GPU measurements, not same-card speedups.')
    lines.append('Missing/OOM/unsupported are not zero scores. Old cropped LongBench results are excluded.')
    write(out/'summary.json',data);(out/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return '\n'.join(lines)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('/data/phasekv-results/a800_quality_fulltext_v1'))
    print(summary(p.parse_args().out))
