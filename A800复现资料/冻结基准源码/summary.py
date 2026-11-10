"""Build a partial or final report without importing any GPU libraries."""
from collections import Counter,defaultdict
from pathlib import Path
import json,os,statistics,sys
from inventory import MAIN,NEW,SHAPES,collect
def read(p):return json.loads(p.read_text(encoding='utf8'))
def summarize(out):
    ids,expected,_=collect();ans={};by_speed={};by_quality={}
    for name in MAIN+NEW:
        home=out/name;gate=home/'gate.json';speed={};quality={}
        for p in sorted((home/'speed').glob('*.json')):
            v=read(p);key=p.stem;d={k:v[k] for k in ('status','batch','shape')}
            if v['status']=='ok':
                for metric in ('aggregate_tokens_per_s','per_sequence_tokens_per_s','ttft_s','prefill_peak_bytes','decode_peak_bytes','peak_allocated_bytes','final_kv_bytes'):
                    d[metric]=statistics.median(r[metric] for r in v['runs'])
                d['input_sha256']=v['input_sha256']
            else:d.update(at=v.get('at'),message=v.get('message'))
            speed[key]=d
        for p in (home/'quality').glob('*.json'):
            v=read(p);quality[v['id']]=v
        by_speed[name]=speed;by_quality[name]=quality
        g=read(gate) if gate.exists() else {}
        ans[name]=dict(gate=g.get('status','not_run'),gate_error=g.get('error'),errors={p.stem:read(p)['error'] for p in home.glob('*_last_error.json')},speed=speed,quality={})
    for name in MAIN+NEW:
        good=defaultdict(list);paired=defaultdict(list);reached=defaultdict(Counter)
        for ident,r in by_quality[name].items():
            group=ids[ident];reached[group][r['status']]+=1
            if r['status']=='ok':
                good[group].append(r['score']);ref=by_quality['full'].get(ident,{})
                if ref.get('status')=='ok' and ref['input_sha256']==r['input_sha256']:paired[group].append([r['score'],ref['score']])
        for group,total in expected.items():
            c=reached[group];pair=paired[group]
            ans[name]['quality'][group]=dict(expected=total,ok=c['ok'],oom=c['oom'],pending=total-sum(c.values()),mean_successful=statistics.mean(good[group]) if good[group] else None,complete=c['ok']==total,paired_n=len(pair),paired_delta_percentage_points=statistics.mean(x-y for x,y in pair) if pair else None)
        for key,x in by_speed[name].items():
            ref=by_speed['full'].get(key,{})
            if x['status']==ref.get('status')=='ok' and x['input_sha256']==ref['input_sha256']:
                x['vs_full_throughput_pct']=100*(x['aggregate_tokens_per_s']/ref['aggregate_tokens_per_s']-1)
    final=dict(protocol='A800 single card, positive throughput delta is faster; same shape, batch and repeat count only',methods=ans)
    out.mkdir(parents=True,exist_ok=True);tmp=out/'summary.tmp';tmp.write_text(json.dumps(final,indent=2),encoding='utf8');tmp.replace(out/'summary.json')
    print('Gates:',{n:v['gate'] for n,v in ans.items()},'Report:',out/'summary.json')
if __name__=='__main__':summarize(Path(sys.argv[1] if len(sys.argv)>1 else os.environ['A800_OUT']))
