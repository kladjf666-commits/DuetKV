"""CPU-only status, paired comparisons, and conservative historical ETA."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
import statistics
import common as c

def mean(values):return statistics.mean(values) if values else None
def macro(records,field='score'):
    tasks=defaultdict(list)
    for r in records:tasks[r['task']].append(r[field])
    return mean([mean(v) for v in tasks.values()])
def fmt(v):return '-' if v is None else f'{v:.2f}'

def report(out,save=True):
    out=Path(out);plan=c.read(out/'plan.json')
    if not plan:return 'Not prepared. Run runner.py prepare first.'
    refs=c.read(out/'references.json')
    if c.digest(refs)!=plan['reference_digest']:raise ValueError('Reference summary changed')
    records=[]
    for row in plan['quality']:
        value=c.quality_record(out,row,allow_error=True)
        records.append(value or dict(row,status='pending'))
    reused=[]
    for row in plan['reused_longgen']:
        path=out/'reused/longgenbench'/c.key(row['id'])
        if c.sha(path)!=row['sha256']:raise ValueError('Reusable result changed')
        reused.append(c.read(path))
    summary=dict(updated=c.now(),experiment=plan['experiment'],profile=c.profile(),suites={},speed=[],
        selection_reused=48,new_quality_target=384,speed_target=16,state=c.read(out/'state.json',{}))
    lines=['PhaseKV: prompt 90%, decode 80%, prompt/cold INT4, recent FP16 128, observation 32.',
           'Quality: full original inputs, B1, natural EOS. Fixed 432-case corpus; LongGenBench 48 reused from selection.',
           'suite          ok   oom error pending  macro    paired Full / old PhaseKV']
    quality_remaining=0.
    for suite in ('longbench','ruler','longgenbench'):
        group=[r for r in records+reused if r['suite']==suite];counts=Counter(r['status'] for r in group)
        good=[r for r in group if r['status']=='ok'];paired={}
        for method in ('full','phasekv'):
            changes=[dict(task=r['task'],score=r['score']-refs['quality'][method][r['id']]['score']) for r in good]
            paired[method]=dict(n=len(changes),macro_delta=macro(changes))
        stat=dict(target=len(group),counts=dict(counts),macro_score=macro(good),paired=paired,
            mean_final_kv_bytes=mean([r['final_kv_bytes'] for r in good]),
            reused_from_selection=(suite=='longgenbench'))
        stat['tasks']={task:dict(ok=sum(r['status']=='ok' for r in group if r['task']==task),
            target=sum(r['task']==task for r in group),mean=mean([r['score'] for r in good if r['task']==task])) for task in sorted({r['task'] for r in group})}
        summary['suites'][suite]=stat
        lines.append(f"{suite:13} {counts['ok']:3} {counts['oom']:5} {counts['error']:5} {counts['pending']:7}  {fmt(stat['macro_score']):>6}  {fmt(paired['full']['macro_delta'])} / {fmt(paired['phasekv']['macro_delta'])}" )
        ratios=[r['wall_s']/refs['quality']['phasekv'][r['id']]['estimate_s'] for r in good if r.get('wall_s') and suite!='longgenbench']
        factor=statistics.median(ratios) if len(ratios)>=3 else 1.
        quality_remaining+=sum(refs['quality']['phasekv'][r['id']]['estimate_s']*factor for r in group if r['status']=='pending')
    speed_remaining=0.;counts=Counter()
    for case in plan['speed']:
        value=c.speed_record(out,case,allow_error=True)
        partial=[c.speed_record(out,case,i,allow_error=True) for i in (1,2,3)]
        status=value.get('status','error' if any(r.get('status')=='error' for r in partial) else 'pending')
        counts[status]+=1
        item=dict(case,status=status,completed_formal=sum(r.get('status')=='ok' for r in partial))
        if status=='ok':
            for k in ('median_total_tps','median_sequence_tps','median_ttft_s','median_final_kv_bytes','max_peak_allocated_bytes'):item[k]=value[k]
            old=refs['speed']['full'][f"{case['shape']}_B{case['batch']}"]
            item['historical_full_tps_ratio']=value['median_total_tps']/old['median_total_tps']
        elif status=='pending':
            old=refs['speed']['phasekv'][f"{case['shape']}_B{case['batch']}"]
            durations=[r['ttft_s']+r['decode_s'] for r in partial if r.get('status')=='ok']
            per=mean(durations) if durations else old['estimate_s']/4
            speed_remaining+=per*(4-item['completed_formal'])
        summary['speed'].append(item)
    summary['speed_counts']=dict(counts)
    summary['complete']=all(r['status'] in ('ok','oom') for r in records) and counts['ok']+counts['oom']==16
    seconds=quality_remaining+speed_remaining
    summary['eta_remaining_seconds']=seconds
    summary['eta_basis']='Historical old PhaseKV per-case wall time; quality calibrated after 3 completed cases per suite. Excludes loading/gates/JIT/restarts; no completion guarantee.'
    eta=datetime.now(timezone(timedelta(hours=8)))+timedelta(seconds=seconds)
    lines.extend([f"Speed: {counts['ok']+counts['oom']}/16 terminal (ok={counts['ok']}, oom={counts['oom']}, error={counts['error']}); warmup 1 + formal 3 per point.",
        'Speed input 8192/32768 x output 256/2048 x B1/2/4/8; actual prompt retention 90%.',
        f"Rough remaining: {seconds/3600:.1f} h; ETA {eta:%Y-%m-%d %H:%M %z}. Loading/gates/JIT excluded; early estimate can be inaccurate.",
        f"State: {summary['state']}",
        'Historical speed ratios use different sessions/possibly different GPUs; old PhaseKV speed retained 100% prompt.',
        'Missing/OOM are not zero scores. LongGenBench is reused parameter-selection evidence, not held-out validation.',
        'COMPLETE' if summary['complete'] else 'INCOMPLETE'])
    if summary['complete']:lines=[x for x in lines if not x.startswith('Rough remaining:')]
    text='\n'.join(lines)
    if save:
        c.write(out/'summary.json',summary)
        (out/'summary.txt').write_text(text+'\n',encoding='utf-8')
    return text
