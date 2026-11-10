"""Two-case A/B retest. Formal generation is unmodified; diagnostics are separate."""
from pathlib import Path
import argparse, hashlib, json, os, platform, signal, sys, time, traceback
import common as c
from runner import inputs, smi
from report import report, validate_pair


def execute(out):
    plan=c.read(out/'plan.json'); cfg=plan['source_config']; experiment=plan['experiment']
    c.verify(c.ROOT,plan['package_manifest']); c.verify(cfg['bench'],c.BASE)
    _,bank=inputs(); session=str(time.time_ns()); gpu=c.read(out/'gpu.json')
    state=dict(phase='loading',session=session,pid=os.getpid(),started=c.now(),current=None)
    io_records=[]
    def save(path,value,*,new=False):
        tick=time.perf_counter(); c.write(path,value,new=new)
        io_records.append(dict(path=str(path.relative_to(out)),seconds=time.perf_counter()-tick,at=c.now()))
        c.write(out/f'checkpoint_io/{session}.json',dict(session=session,records=io_records,
                note='Measured checkpoint calls occur outside generation/diagnostic timing. Journal self-writes are excluded.'))
    def update(**values):
        state.update(values,updated=c.now()); save(out/'state.json',state)
    def interrupt(*_): raise KeyboardInterrupt('Completed A/B pairs retained; incomplete pairs are rerun together')
    signal.signal(signal.SIGTERM,interrupt); signal.signal(signal.SIGINT,interrupt)
    engine=None
    try:
        update()
        sys.path.append(cfg['bench'])
        import torch,transformers,flash_attn,triton
        from engine import Engine,run as native_run
        import native_runtime as runtime
        from baseline_runtime import make_engine_class,check_measurement,synthetic_gate
        from PhaseKV.memory_prefill import memory_prefill
        from stage_probe import StageProbe
        import PhaseKV
        if Path(PhaseKV.__file__).resolve().parent != c.ROOT/'PhaseKV': raise RuntimeError('Wrong PhaseKV source')
        versions=dict(torch=str(torch.__version__),transformers=transformers.__version__,flash=flash_attn.__version__,
                      triton=triton.__version__,python=platform.python_version(),omp_threads=os.environ['OMP_NUM_THREADS'])
        if any(v!=plan['source_protocol'][k] for k,v in versions.items()): raise RuntimeError('Preserved environment changed')
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(8,0) or 'A800' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expected one visible SM80 A800')
        torch.manual_seed(20261029)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        protocol=dict(experiment=experiment,gpu=gpu,versions=versions,model=cfg['model'],weights=cfg['weights'],
            profile=c.PROFILE,effective_options=c.options(c.PROFILE),seed=20261029,tf32=False,
            cases=c.CASES,formal_rounds=3,diagnostic_pairs_per_case=1,
            description='Same loaded model and physical GPU. Formal timing uses original native run and a plain dict. '
                        'Instrumented prefill-only diagnostics are separate and excluded from formal statistics.')
        protocol=json.loads(json.dumps(protocol))
        if (out/'protocol.json').exists():
            if c.read(out/'protocol.json')!=protocol: raise ValueError('Protocol changed')
        else: save(out/'protocol.json',protocol,new=True)

        class PairedEngine(make_engine_class(Engine,runtime)):
            variant='baseline'
            def start(self,ids,limit,quality,progress):
                if self.variant=='baseline': return super().start(ids,limit,quality,progress)
                self.offset=ids.shape[1]; progress['stage']='prefill'
                self.effective,self.mode=runtime.reset_sample(self.model,'phasekv',self.options,limit)
                logits,self.cache=memory_prefill(self.model,ids,self.options,runtime,progress,prefix_chunk_size=512)
                return logits

        def telemetry():
            snapshot=dict(at=c.now(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),
                          load_average=list(os.getloadavg()),torch_threads=torch.get_num_threads())
            apps=smi('--query-compute-apps=gpu_uuid,pid')
            foreign=[pid for uuid,pid in apps if uuid==gpu['uuid'] and int(pid)!=os.getpid()]
            if foreign: raise RuntimeError('Other compute process appeared on selected GPU: '+str(foreign))
            snapshot['compute_processes']=[dict(uuid=u,pid=p) for u,p in apps if u==gpu['uuid']]
            try:
                fields='uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,memory.used'
                rows=smi('--query-gpu='+fields)
                snapshot['gpu']=next(dict(zip(fields.split(','),row)) for row in rows if row[0]==gpu['uuid'])
            except Exception as exc: snapshot['telemetry_error']=repr(exc)
            return snapshot

        def packed_hash(cache):
            h=hashlib.sha256()
            for layer in cache.layers:
                for kv in (layer.prompt.key,layer.prompt.value):
                    for tensor in (kv.packed,kv.scale,kv.mins):
                        h.update(str((tuple(tensor.shape),str(tensor.dtype))).encode())
                        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        def gates():
            checks=[]
            for batch in (1,8):
                pair=[]; ids=torch.tensor([r[:2112] for r in bank[:batch]],device='cuda')
                for arm in c.ARMS:
                    engine.variant=arm
                    logits=engine.start(ids,164,False,{})
                    first=logits.detach().cpu(); packed=packed_hash(engine.cache)
                    token=logits[:,-1].argmax(-1,keepdim=True); generated=[token]
                    for _ in range(163):
                        logits=engine.step(token);token=logits[:,-1].argmax(-1,keepdim=True);generated.append(token)
                    evidence=dict(engine.evidence(),effective_options=engine.effective,batch=batch,decode_steps=163)
                    check_measurement(evidence,c.PROFILE,2112)
                    pair.append((first,packed,c.digest(torch.cat(generated,dim=1).tolist())))
                    engine.release()
                torch.testing.assert_close(pair[0][0],pair[1][0],rtol=0,atol=0)
                if pair[0][1:]!=pair[1][1:]: raise ValueError('Gate packed KV/generated tokens differ')
                checks.append(dict(batch=batch,packed_identical=True,logits_identical=True,generated_identical=True))
            return dict(experiment=experiment,session=session,gpu_uuid=gpu['uuid'],status='ok',checks=checks,
                        synthetic_b8=synthetic_gate(c.options(c.PROFILE),batch=8))

        def measure(arm,case,diagnostic=False):
            n,cap,batch=case; rows=[r[:n] for r in bank[:batch]]
            engine.variant=arm
            update(phase='diagnostic' if diagnostic else 'measurement',current=dict(case=case,arm=arm))
            before=telemetry()
            try:
                if not diagnostic:
                    # Plain dict: no status callbacks, profiler, event probes or subprocesses during run.
                    with c.timed_region(): result=native_run(engine,rows,cap,[],False,{})
                    output=result.pop('output_ids')
                    result.update(output_sha256=c.digest(output),first_token_sha256=c.digest([r[:1] for r in output]))
                else:
                    ids=torch.tensor(rows,dtype=torch.long,device='cuda')
                    probe=StageProbe(runtime,n,c.PROFILE['obs_window'])
                    with probe:
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        with c.timed_region():
                            start=time.perf_counter();logits=engine.start(ids,cap,False,{})
                            with probe.span('first_token'): token=logits[:,-1].argmax(-1,keepdim=True)
                            torch.cuda.synchronize();ttft=time.perf_counter()-start
                        peak=torch.cuda.max_memory_allocated()
                    result=dict(batch=batch,input_tokens=n,output_tokens_per_row=1,decode_steps=0,ttft_s=ttft,
                        prefill_peak_bytes=peak,peak_allocated_bytes=peak,effective_options=engine.effective,
                        first_token_sha256=c.digest(token.tolist()),packed_prompt_sha256=packed_hash(engine.cache),
                        phase_timings=probe.result(),**engine.evidence())
                    del logits,token,ids
                check_measurement(result,c.PROFILE,n)
                if not diagnostic and result['output_tokens_per_row']!=cap: raise ValueError('Wrong output length')
                result.update(status='ok',arm=arm,gpu_uuid=gpu['uuid'],
                    input_sha256=[hashlib.sha256(json.dumps(row).encode()).hexdigest() for row in rows],
                    telemetry_before=before,telemetry_after=telemetry())
                return result
            finally: engine.release()

        with torch.inference_mode():
            with c.lock('/tmp/phasekv-a800-model-load.lock'):
                engine=PairedEngine('phasekv',Path(cfg['model']))
            engine.configure(c.PROFILE)
            update(phase='gates'); telemetry()
            save(out/f'gates/{session}.json',gates(),new=True)
            for case_index,case in enumerate(c.CASES):
                name=c.case_name(case)
                pending=[i for i in (1,2,3) if not (out/f'pairs/{name}/r{i}.json').exists()]
                if not pending and (out/f'diagnostics/{name}.json').exists(): continue
                for arm in c.arm_order(case_index,1):
                    value=measure(arm,case)
                    save(out/f'warmups/{session}/{name}/{arm}.json',value,new=True)
                for index in pending:
                    order=c.arm_order(case_index,index); values={}
                    started=c.now()
                    for arm in order: values[arm]=measure(arm,case)
                    pair=dict(experiment=experiment,session=session,gpu_uuid=gpu['uuid'],case=list(case),
                        status='ok',kind='formal',formal_round=index,order=list(order),arms=values,
                        started=started,finished=c.now())
                    validate_pair(out,pair,experiment,case,'formal',index)
                    save(out/f'pairs/{name}/r{index}.json',pair,new=True)
                    report(out)
                    print(c.now(),name,'pair',index,'TTFT baseline/optimized',
                          values['baseline']['ttft_s'],values['optimized']['ttft_s'],flush=True)
            # All headline formal measurements finish before any timing instrumentation.
            for case_index,case in enumerate(c.CASES):
                name=c.case_name(case); path=out/f'diagnostics/{name}.json'
                if path.exists(): continue
                order=c.arm_order(case_index,2); values={arm:measure(arm,case,True) for arm in order}
                pair=dict(experiment=experiment,session=session,gpu_uuid=gpu['uuid'],case=list(case),
                          status='ok',kind='diagnostic',order=list(order),arms=values)
                validate_pair(out,pair,experiment,case,'diagnostic')
                save(path,pair,new=True);report(out)
                print(c.now(),name,'diagnostics complete',flush=True)
        if not report(out)['complete']: raise RuntimeError('Incomplete campaign')
        update(phase='complete',current=None)
        print('COMPLETE',c.now(),flush=True)
    except BaseException as exc:
        update(phase='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if engine is not None: engine.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    execute(parser.parse_args().out)
