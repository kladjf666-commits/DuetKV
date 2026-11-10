from pathlib import Path
import argparse
import gc
import os
import sys
import time
import traceback
from common import HERE,read,write,rows,path_for,result,terminal,sha,verify,weight_identity

def execute(args):
    config=read(args.config);out=Path(config['out']);method=args.method;dest=out/'results'/method
    if verify(HERE)!=config['manifest']:raise ValueError('Recovery package changed')
    if weight_identity(config['model'])!=config['weights']:raise ValueError('Model files changed')
    sys.path.insert(0,config['bench'])
    import torch,transformers,flash_attn,triton,fcntl
    from engine import Engine,run
    from worker import gate,hw
    from benchmark_metrics import score
    from transformers import AutoTokenizer
    gpu=hw()
    expected=('2.5.0','4.45.2') if method=='clusterkv' else ('2.5.1','4.46.3')
    if torch.__version__.split('+')[0]!=expected[0] or transformers.__version__!=expected[1]:raise ValueError('Unexpected existing environment')
    if flash_attn.__version__!='2.7.4.post1' or triton.__version__!='3.1.0':raise ValueError('Native dependency versions changed')
    protocol=dict(manifest=config['manifest'],benchmark_manifest=config['bench_manifest'],method=method,hardware='A800 80GB SM80',
        model=config['model'],weights=config['weights'],torch=str(torch.__version__),transformers=transformers.__version__,
        flash=flash_attn.__version__,triton=triton.__version__,input_policy='full original document, no truncation, B1 natural EOS',
        corpus_sha256=sha(HERE/'data/fulltext.jsonl.gz'),threads=config['threads'],quality_prompt_pct=95,
        kivi_fp16_reduced_precision_reduction=False if method=='kivi' else None)
    if read(dest/'protocol.json',protocol)!=protocol:raise ValueError('Quality protocol changed')
    write(dest/'protocol.json',protocol)
    write(dest/'gpu_sessions'/f'{time.time_ns()}.json',dict(gpu=gpu,stage=args.stage,pid=os.getpid()))
    selected=[r for r in rows() if args.stage=='all' or r['pilot']]
    todo=[r for r in selected if not terminal(result(out,method,r))]
    # Respect the native validated profile; never shorten input to meet its bound.
    for row in list(todo):
        clusters=max(1,len(row['input_ids'])//80)+((row['max_new_tokens']-1)//320)*4
        if method=='clusterkv' and clusters>512:
            write(path_for(out,method,row),dict(status='unsupported',id=row['id'],suite='longbench',task=row['task'],
                input_sha256=row['input_sha256'],input_tokens=len(row['input_ids']),truncated_input=False,
                reason='Native validated ClusterKV profile requires at most 512 clusters',required_clusters=clusters,gpu=gpu))
            todo.remove(row)
    if not todo:return
    torch.manual_seed(20261029);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if method=='kivi':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    with open('/tmp/phasekv-a800-model-load.lock','a') as load_lock:
        fcntl.flock(load_lock,fcntl.LOCK_EX)
        engine=Engine(method,Path(config['model']))
    if read(dest/'gate.json',{}).get('status')!='ok':
        import gzip,json
        with gzip.open(Path(config['bench'])/'data/speed_rows.jsonl.gz','rt') as f:bank=[json.loads(next(f))['input_ids']]
        write(dest/'gate.json',dict(status='running'))
        details=gate(engine,bank,1)
        if method=='phasekv':
            from phase_gate import check_phase
            details['a800_phase']=check_phase(engine,bank[0])
        write(dest/'gate.json',dict(status='ok',source='new B1 GPU gate',**details))
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    for row in todo:
        path=path_for(out,method,row)
        if path.exists():
            backup=dest/'previous_errors'/f'{path.stem}.{time.time_ns()}.json';backup.parent.mkdir(parents=True,exist_ok=True)
            path.replace(backup)
        progress={};start=time.time()
        write(dest/'current.json',dict(id=row['id'],input_tokens=len(row['input_ids']),stage='running',time=start))
        record=dict(id=row['id'],suite='longbench',task=row['task'],input_sha256=row['input_sha256'],
            input_tokens=len(row['input_ids']),max_new_tokens=row['max_new_tokens'],truncated_input=False,
            original_id=row['original_id'],evaluation_split=row['evaluation_split'],gpu=gpu)
        try:
            value=run(engine,[row['input_ids']],row['max_new_tokens'],set(row['stop_ids']),True,progress)
            ids=value.pop('output_ids')[0];text=tokenizer.decode(ids,skip_special_tokens=True)
            record.update(value,status='ok',output_ids=ids,prediction=text,answers=row['answers'],**score(row,text))
            record['stopped_on_eos']=bool(ids and ids[-1] in row['stop_ids'])
        except torch.cuda.OutOfMemoryError as exc:
            record.update(status='oom',message=str(exc),at=progress.copy(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
        except Exception as exc:
            record.update(status='error',message=repr(exc),at=progress.copy(),traceback=traceback.format_exc())
        record['wall_s']=time.time()-start;write(path,record)
        engine.release();gc.collect()
        print(row['id'],record['status'],record.get('score'),flush=True)
        if record['status']=='error':raise RuntimeError('Saved native runtime error; stop this method for review')
    write(dest/'current.json',dict(stage='done',time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--method',required=True)
    p.add_argument('--stage',choices=('pilot','all'),required=True);args=p.parse_args()
    try:
        import torch
        with torch.inference_mode():execute(args)
        cfg=read(args.config);(Path(cfg['out'])/'results'/args.method/'last_error.json').unlink(missing_ok=True)
    except Exception as exc:
        cfg=read(args.config)
        write(Path(cfg['out'])/'results'/args.method/'last_error.json',dict(status='error',message=repr(exc),traceback=traceback.format_exc()))
        raise
