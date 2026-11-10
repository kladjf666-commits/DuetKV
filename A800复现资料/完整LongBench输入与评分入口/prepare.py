from pathlib import Path
import argparse
import hashlib
import shutil
from common import BASE_MANIFEST,HERE,METHODS,SHAPES,read,rows,sha,verify,weight_identity,write,path_for,terminal,result,speed_matrix

def compatible(protocol,config,method):
    torch,transformers=('2.5.0','4.45.2') if method=='clusterkv' else ('2.5.1','4.46.3')
    return bool(protocol and protocol.get('manifest')==BASE_MANIFEST and protocol.get('method')==method
                and protocol.get('weights')==config['weights'] and Path(protocol.get('model','')).resolve()==Path(config['model']).resolve()
                and protocol.get('quality_prompt_pct')==95 and protocol.get('torch','').split('+')[0]==torch
                and protocol.get('transformers')==transformers and protocol.get('flash')=='2.7.4.post1' and protocol.get('triton')=='3.1.0'
                and 'A800' in protocol.get('gpu',{}).get('gpu','')
                and (method!='kivi' or protocol.get('kivi_fp16_reduced_precision_reduction') is False))

def reuse(config):
    out=Path(config['out']);old=Path(config['old_out'])/'results';counts={}
    for method in METHODS:
        protocol=read(old/method/'protocol.json',{});gate=read(old/method/'gate.json',{})
        counts[method]=0
        if not compatible(protocol,config,method) or gate.get('status')!='ok':continue
        source_prot=sha(old/method/'protocol.json')
        write(out/'results'/method/'inherited_protocol.json',protocol)
        if not (out/'results'/method/'gate.json').exists():
            write(out/'results'/method/'gate.json',dict(status='ok',source='existing same-model/same-code A800 gate',
                original_gate=gate,original_gpu=protocol['gpu'],original_gate_sha256=sha(old/method/'gate.json')))
        for row in rows():
            if terminal(result(out,method,row)):continue
            for alias in row['reuse_aliases']:
                source=old/method/'quality'/(hashlib.sha256(alias['id'].encode()).hexdigest()+'.json')
                record=read(source,{})
                if record.get('status') not in ('ok','oom'):continue
                if record.get('id')!=alias['id'] or record.get('input_sha256')!=row['input_sha256']:
                    raise ValueError('Existing quality record does not match full input: '+str(source))
                if record['status']=='ok' and (record['output_tokens_per_row']>row['max_new_tokens'] or record['input_tokens']!=len(row['input_ids'])):
                    raise ValueError('Existing result shape differs')
                copy=dict(record,id=row['id'],suite='longbench',task=row['task'],input_sha256=row['input_sha256'],
                    truncated_input=False,max_new_tokens=row['max_new_tokens'],answers=row['answers'],
                    reused_from=dict(path=str(source),id=alias['id'],sha256=sha(source),protocol_sha256=source_prot))
                write(path_for(out,method,row),copy);counts[method]+=1;break
    write(out/'reuse_summary.json',dict(new_this_scan=counts,total={m:sum(bool(result(out,m,r).get('reused_from')) for r in rows()) for m in METHODS}))
    return counts

def prepare(old_out,kit,out):
    identity=verify(HERE);verify(kit)
    old_out=old_out.resolve();out=out.resolve();kit=kit.resolve()
    if out==old_out or out.is_relative_to(old_out) or old_out.is_relative_to(out):raise ValueError('Use a separate sibling output')
    layout=read(old_out/'layout.json',{});bench=Path(layout['bench']).resolve()
    if verify(bench)!=BASE_MANIFEST:raise ValueError('Unexpected original benchmark')
    if layout['kit_manifest']!=sha(kit/'SHA256SUMS'):raise ValueError('Original kit differs')
    queue=read(old_out/'queue.json',{})
    if len(queue.get('identity',{}).get('gpus',{}))!=4:raise ValueError('Expected existing four-GPU queue')
    if queue['identity']['kit']!=sha(kit/'SHA256SUMS'):raise ValueError('Queue belongs to another kit')
    config=dict(manifest=identity,bench_manifest=BASE_MANIFEST,old_out=str(old_out),kit=str(kit),out=str(out),
        bench=str(bench),model=layout['model'],envs=layout['envs'],weights=weight_identity(layout['model']),
        threads=queue['identity']['threads'],gpus=queue['identity']['gpus'],speed_batches=[1,2,4,8],speed_shapes=list(SHAPES),quality_batch=1)
    for group in ('main','million','clusterkv'):
        if not (Path(config['envs'])/group/'bin/python').is_file():raise ValueError('Missing existing environment: '+group)
    existing=read(out/'config.json')
    if existing is not None and existing!=config:raise ValueError('Recovery configuration changed; select a new output')
    if out.exists() and any(out.iterdir()) and not existing:raise ValueError('Output is not empty and unrecognized')
    out.mkdir(parents=True,exist_ok=True)
    write(out/'config.json',config)
    write(out/'speed_matrix.json',dict(cases=speed_matrix(str(bench)),batches=[1,2,4,8],b1_only=['million','clusterkv'],points=104,
        measured_rounds=3,warmup_rounds=1,execution='four independent single-GPU workers; per-method physical GPU preserved'))
    for name in ('key.pt','value.pt','meta.json'):
        source=old_out/'results/codebook'/name;target=out/'results/codebook'/name
        if target.exists() and sha(source)!=sha(target):raise ValueError('Codebook changed')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(source,target)
    reused=reuse(config)
    print('PREPARED:',out,'newly reused results:',reused,flush=True)
    return config

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-out',type=Path,default=Path('/data/phasekv-results/a800_four_gpu_v1'))
    p.add_argument('--kit',type=Path,default=Path('/data/a800_four_gpu_v1'))
    p.add_argument('--out',type=Path,default=Path('/data/phasekv-results/a800_quality_fulltext_v1'))
    a=p.parse_args();prepare(a.old_out,a.kit,a.out)
