"""One loaded model, immutable per-case/per-round results, full-input quality."""
from pathlib import Path
import argparse,os,platform,signal,sys,time,traceback
import common as c
from report import report
from profile_runtime import make_engine_class,check_measurement,gate,batch_gate

def execute(out,uuid):
    plan=c.read(out/'plan.json');cfg=plan['source_config'];p=c.profile()
    c.verify(c.HERE,plan['package_manifest']);c.verify(cfg['bench'],c.BASE)
    rows=c.quality_rows(cfg);cases=c.speed_cases(cfg);bank=c.speed_bank(cfg)
    if [dict(c.meta(r),pilot=r.get('pilot',False)) for r in rows]!=plan['quality'] or cases!=plan['speed']:raise ValueError('Frozen input plan changed')
    state=dict(started=c.now(),phase='loading_environment',gpu_uuid=uuid,pid=os.getpid(),current=None)
    def update(**kw):state.update(kw,updated=c.now());c.write(out/'state.json',state)
    class Progress(dict):
        last=0.
        def __setitem__(self,key,value):
            changed=key=='stage' and self.get(key)!=value
            super().__setitem__(key,value)
            if changed or time.monotonic()-self.last>10:
                self.last=time.monotonic();update(at=dict(self))
    def stop(signum,frame):raise KeyboardInterrupt('Worker stopped; committed results retained')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);update()
    engine=None
    try:
        sys.path.insert(0,cfg['bench'])
        import torch,transformers,flash_attn,triton
        from engine import Engine,run as native_run
        import native_runtime as rt
        from worker import hw,quality_score,oom_memory
        from transformers import AutoTokenizer
        gpu=hw();old=plan['source_protocol']
        if gpu['uuid'].removeprefix('GPU-')!=uuid.removeprefix('GPU-'):raise ValueError('Actual GPU differs')
        versions=dict(torch=str(torch.__version__),transformers=transformers.__version__,flash=flash_attn.__version__,triton=triton.__version__,python=platform.python_version(),omp_threads=os.environ['OMP_NUM_THREADS'])
        if any(versions[k]!=old[k] for k in versions):raise ValueError('Preserved environment changed: '+str(versions))
        protocol=dict(experiment=plan['experiment'],gpu=gpu,versions=versions,weights=cfg['weights'],model=cfg['model'],
            benchmark_manifest=c.BASE,package_manifest=plan['package_manifest'],profile=p,effective_options=c.options(p),
            seed=20261029,tf32=False,quality='320 complete-input LongBench + 64 RULER32K, B1, greedy, original EOS/caps',
            reuse='48 exact profile LongGenBench selection cases; byte-identical source records',
            speed='16 points, 1 warmup + 3 formal; per-request prompt90 selection included in TTFT; native batched decode')
        saved=c.read(out/'protocol.json')
        if saved and saved!=protocol:raise ValueError('Protocol changed')
        if not saved:c.write(out/'protocol.json',protocol,new=True)
        torch.manual_seed(20261029);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def validate(name,fn):
            gp=out/'gates'/f'{name}.json';saved=c.read(gp,{})
            if saved and (saved.get('experiment')!=plan['experiment'] or saved.get('profile')!=p or saved.get('gpu')!=gpu):raise ValueError('Gate identity changed')
            if saved.get('status')=='ok':return
            update(phase='validating_'+name,current=None,at=None)
            try:c.write(gp,dict(fn(),experiment=plan['experiment'],profile=p,gpu=gpu))
            except BaseException as exc:
                c.write(gp,dict(status='failed',experiment=plan['experiment'],profile=p,gpu=gpu,error=repr(exc),traceback=traceback.format_exc()));raise
        def quality(row):
            if c.quality_record(out,row):return
            started=time.monotonic();progress=Progress()
            update(phase='quality_'+row['suite'],current=row['id'],input_tokens=len(row['input_ids']),max_new_tokens=row['max_new_tokens'],at={'stage':'prefill'})
            record=dict(c.meta(row),experiment=plan['experiment'],profile=p,gpu=gpu,batch=1,truncated_input=False,
                measurement_role='evaluation',source='new_inference',pilot=row.get('pilot',False),stop_ids=row['stop_ids'])
            for k in ('original_id','source_index','source_context_sha256','prompt_sha256','evaluation_split','ruler_base_task','length_budget'):
                if k in row:record[k]=row[k]
            try:
                value=native_run(engine,[row['input_ids']],row['max_new_tokens'],set(row['stop_ids']),True,progress)
                check_measurement(value,p,len(row['input_ids']))
                ids=value.pop('output_ids')[0];text=tokenizer.decode(ids,skip_special_tokens=True)
                record.update(value,status='ok',output_ids=ids,prediction=text,answers=row['answers'],
                    stopped_on_eos=bool(ids and ids[-1] in row['stop_ids']),**quality_score(row,text))
            except torch.cuda.OutOfMemoryError as exc:record.update(status='oom',message=str(exc),at=dict(progress),**oom_memory())
            except Exception as exc:record.update(status='error',message=repr(exc),at=dict(progress),traceback=traceback.format_exc())
            record['wall_s']=time.monotonic()-started
            c.write(c.quality_path(out,row),record,new=True);engine.release()
            update(last_completed=row['id'],current=None,at=None);report(out)
            print(c.now(),row['suite'],row['id'],record['status'],record.get('score'),flush=True)
            if record['status']=='error':raise RuntimeError('Saved quality runtime error; stop for review')
        def speed(case):
            if c.finish_speed(out,case):return
            indices=[i for i in (1,2,3) if not c.speed_record(out,case,i)]
            inputs=[r[:case['input_tokens']] for r in bank[:case['batch']]]
            # Each invocation warms the point, including after a restart. Only missing formal rounds run.
            for index in [0]+indices:
                started=time.monotonic();progress=Progress()
                label=f"{case['shape']}_B{case['batch']}_"+('warmup' if index==0 else f'r{index}')
                update(phase='speed',current=label,at=None)
                record=dict(case,experiment=plan['experiment'],profile=p,gpu=gpu,formal_round=index,
                    measurement_role='warmup' if index==0 else 'speed',truncated_input=False)
                try:
                    value=native_run(engine,inputs,case['max_new_tokens'],[],False,progress)
                    check_measurement(value,p,case['input_tokens'])
                    if value['output_tokens_per_row']!=case['max_new_tokens']:raise ValueError('Incomplete speed output')
                    value['output_sha256']=c.digest(value.pop('output_ids'));record.update(value,status='ok')
                except torch.cuda.OutOfMemoryError as exc:record.update(status='oom',message=str(exc),at=dict(progress),**oom_memory())
                except Exception as exc:record.update(status='error',message=repr(exc),at=dict(progress),traceback=traceback.format_exc())
                record['wall_s']=time.monotonic()-started;engine.release()
                # A warmup failure terminates the point without pretending a timed round succeeded.
                # Store in the first *missing* checkpoint; existing formal rounds are never touched.
                if index or record['status']!='ok':c.write(c.speed_path(out,case,index or indices[0]),record,new=True)
                else:
                    warm=out/'warmups'/f"{case['shape']}_B{case['batch']}_{time.time_ns()}.json";c.write(warm,record,new=True)
                update(last_completed=label,current=None,at=None)
                print(c.now(),label,record['status'],record.get('aggregate_tokens_per_s'),flush=True)
                if record['status']=='error':report(out);raise RuntimeError('Saved speed runtime error; stop for review')
                if record['status']=='oom':break
            c.finish_speed(out,case);report(out)
        update(phase='loading_model')
        with torch.inference_mode():
            with c.lock('/tmp/phasekv-a800-model-load.lock'):engine=make_engine_class(Engine,rt)('phasekv',Path(cfg['model']))
            engine.configure(p);tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
            validate('quality',lambda:gate(engine,dict(input_ids=bank[0]),p,native_run))
            for row in rows:
                if row.get('pilot'):quality(row)
            if any(not c.finish_speed(out,case) for case in cases):
                validate('speed_batches',lambda:batch_gate(engine,bank,p,native_run))
                for case in cases:speed(case)
            for row in rows:
                if not row.get('pilot'):quality(row)
            c.preserved(c.read(out/'preserved.json'))
        update(phase='complete',current=None,at=None);print(report(out),flush=True)
    except BaseException as exc:
        update(phase='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--uuid',required=True)
    a=p.parse_args();execute(a.out,a.uuid)
