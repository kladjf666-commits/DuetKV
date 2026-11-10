"""One model, same-GPU paired arms; immutable per-round checkpoints."""
from pathlib import Path
import argparse, hashlib, os, platform, signal, sys, time, traceback
import common as c
from runner import inputs
from report import report


def execute(out):
    plan = c.read(out / 'plan.json'); cfg = plan['source_config']; experiment = plan['experiment']
    c.verify(c.ROOT, plan['package_manifest']); c.verify(cfg['bench'], c.BASE)
    quality, bank = inputs()
    state = dict(phase='loading', current=None, started=c.now(), pid=os.getpid())
    def update(**values):
        state.update(values, updated=c.now()); c.write(out / 'state.json', state)
    def interrupt(*_): raise KeyboardInterrupt('Committed results retained')
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    update()
    engine = None
    try:
        # Keep this benchmark's private engine ahead of the preserved runtime.
        sys.path.append(cfg['bench'])
        import torch, transformers, flash_attn, triton
        from engine import Engine, run as native_run
        import native_runtime as runtime
        import benchmark_metrics
        from baseline_runtime import make_engine_class, check_measurement, synthetic_gate
        from PhaseKV.memory_prefill import memory_prefill
        import PhaseKV
        if Path(PhaseKV.__file__).resolve().parent != c.ROOT / 'PhaseKV':
            raise RuntimeError('Wrong PhaseKV import')
        versions = dict(torch=str(torch.__version__), transformers=transformers.__version__,
                        flash=flash_attn.__version__, triton=triton.__version__,
                        python=platform.python_version(), omp_threads=os.environ['OMP_NUM_THREADS'])
        if any(v != plan['source_protocol'][k] for k, v in versions.items()):
            raise RuntimeError('Preserved environment changed: ' + str(versions))
        gpu = c.read(out / 'gpu.json'); props = torch.cuda.get_device_properties(0)
        if 'A800' not in props.name or torch.cuda.get_device_capability(0) != (8, 0):
            raise RuntimeError('Actual device is not SM80 A800')
        protocol = dict(experiment=experiment, gpu=gpu, versions=versions, profile=c.PROFILE,
                        effective_options=c.options(c.PROFILE), model=cfg['model'], weights=cfg['weights'],
                        seed=20261029, tf32=False, cases=c.CASES,
                        description='Same loaded model; original P90 prefill vs bounded-lifetime prefill; unchanged decode kernels')
        path = out / 'protocol.json'
        # Normalize tuples before comparing serialized evidence.
        protocol = __import__('json').loads(__import__('json').dumps(protocol))
        if path.exists():
            if c.read(path) != protocol: raise ValueError('Protocol changed')
        else: c.write(path, protocol, new=True)
        torch.manual_seed(20261029)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

        class PairedEngine(make_engine_class(Engine, runtime)):
            variant = 'baseline'
            def start(self, ids, limit, quality, progress):
                if self.variant == 'baseline': return super().start(ids, limit, quality, progress)
                if quality and ids.shape[0] != 1: raise ValueError('Quality requires B1')
                self.offset = ids.shape[1]; progress['stage'] = 'prefill'
                self.effective, self.mode = runtime.reset_sample(self.model, 'phasekv', self.options, limit)
                logits, self.cache = memory_prefill(self.model, ids, self.options, runtime, progress,
                                                   prefix_chunk_size=None if quality else 512)
                return logits

        with c.lock('/tmp/phasekv-a800-model-load.lock'):
            engine = PairedEngine('phasekv', Path(cfg['model']))
        engine.configure(c.PROFILE)
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True)

        def existing(path):
            if path.exists(): return c.validate_saved(c.read(path), experiment)
            return None

        class Progress(dict):
            last = 0.
            def __setitem__(self, key, value):
                changed = key == 'stage' and self.get(key) != value
                super().__setitem__(key, value)
                if changed or time.monotonic() - self.last > 10:
                    self.last = time.monotonic(); update(at=dict(self))

        def measure(path, arm, rows, cap, stops, natural, extra):
            engine.variant = arm; update(current=str(path.relative_to(out)), at=None)
            record = dict(experiment=experiment, arm=arm, gpu_uuid=gpu['uuid'], **extra)
            try:
                result = native_run(engine, rows, cap, stops, natural, Progress())
                check_measurement(result, c.PROFILE, len(rows[0]))
                if not natural and result['output_tokens_per_row'] != cap:
                    raise ValueError('Speed output length changed')
                output = result.pop('output_ids'); result['output_sha256'] = c.digest(output)
                if natural:
                    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
                    result.update(output_ids=output[0], prediction=prediction,
                                  stopped_on_eos=output[0][-1] in stops,
                                  **benchmark_metrics.score(extra['sample'], prediction))
                    record.pop('sample')
                record.update(result, status='ok')
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)): raise
                record.update(status='oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'error',
                              error=repr(exc), traceback=traceback.format_exc())
                c.write(path, record, new=True)
                raise
            finally:
                engine.release()
            c.write(path, record, new=True)
            print(c.now(), path.relative_to(out), 'ok',
                  'peak_GiB', record['peak_allocated_bytes'] / 2**30,
                  'decode_tps', record['aggregate_tokens_per_s'], flush=True)
            report(out)
            return record

        def packed_hash(cache):
            h = hashlib.sha256()
            for layer in cache.layers:
                for kv in (layer.prompt.key, layer.prompt.value):
                    for tensor in (kv.packed, kv.scale, kv.mins):
                        h.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
                        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
            return h.hexdigest()

        def gates():
            checks = []
            for natural, batch in ((True, 1), (False, 1), (False, 8)):
                pair = []
                ids = torch.tensor([row[:2112] for row in bank[:batch]], device='cuda')
                for arm in c.ARMS:
                    engine.variant = arm
                    logits = engine.start(ids, 164, natural, {})
                    first = logits.detach().cpu(); cache_hash = packed_hash(engine.cache)
                    token = logits[:, -1].argmax(-1, keepdim=True); generated = [token]
                    for _ in range(163):
                        logits = engine.step(token); token = logits[:, -1].argmax(-1, keepdim=True)
                        generated.append(token)
                    evidence = dict(engine.evidence(), effective_options=engine.effective,
                                    batch=batch, decode_steps=163)
                    check_measurement(evidence, c.PROFILE, 2112)
                    pair.append((first, cache_hash, c.digest(torch.cat(generated, dim=1).tolist())))
                    engine.release()
                torch.testing.assert_close(pair[0][0], pair[1][0], rtol=0, atol=0)
                if pair[0][1:] != pair[1][1:]:
                    raise ValueError('A/B packed prompt or generated token mismatch')
                checks.append(dict(quality=natural, batch=batch, prompt_tokens=2112,
                                   generated_tokens=164, packed_prompt_identical=True,
                                   first_logits_identical=True, generated_tokens_identical=True))
            return dict(experiment=experiment, status='ok', checks=checks,
                        synthetic_b8=synthetic_gate(c.options(c.PROFILE), batch=8))

        with torch.inference_mode():
            update(phase='gates', current='gates')
            if not existing(out / 'gates.json'): c.write(out / 'gates.json', gates(), new=True)
            for case in c.CASES:
                n, cap, batch = case; name = c.case_name(case); rows = [r[:n] for r in bank[:batch]]
                update(phase='speed')
                pending = c.pending_rounds(out, case, experiment)
                # Rewarm only arms with uncommitted rounds after interruption.
                for arm in c.ARMS:
                    if pending[arm]:
                        measure(out / f'warmups/{name}/{arm}/{time.time_ns()}.json', arm,
                                rows, cap, [], False, dict(role='warmup', case=case))
                for i in (1, 2, 3):
                    for arm in (c.ARMS[::-1] if i % 2 else c.ARMS):
                        if i in pending[arm]:
                            measure(out / f'speed/{name}/{arm}/r{i}.json', arm, rows, cap, [], False,
                                    dict(role='speed', case=case, formal_round=i))
                    result = report(out)
                    if not all(result['output_matches'].values()):
                        raise ValueError('Paired speed outputs diverged; stop for review')
            update(phase='quality')
            for sample in quality:
                for arm in c.ARMS:
                    path = out / f'quality/{c.digest(sample["id"])}/{arm}.json'
                    if not existing(path):
                        measure(path, arm, [sample['input_ids']], sample['max_new_tokens'],
                                sample['stop_ids'], True, dict(role='quality', id=sample['id'],
                                input_sha256=sample['input_sha256'], suite=sample['suite'], task=sample['task'],
                                max_new_tokens=sample['max_new_tokens'], stop_ids=sample['stop_ids'],
                                truncated_input=False, sample=sample))
                if not all(report(out)['output_matches'].values()):
                    raise ValueError('Paired quality outputs diverged; stop for review')
        if not report(out)['complete']: raise RuntimeError('Incomplete results')
        update(phase='complete', current=None, at=None)
        report(out); print('COMPLETE', c.now(), flush=True)
    except BaseException as exc:
        update(phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
               error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if engine is not None: engine.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    execute(parser.parse_args().out)
