from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib, json, sys, tempfile, time, unittest
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'benchmark'),str(ROOT)]
import common as c
import baseline_runtime as baseline
from stage_probe import StageProbe
from report import report, validate_pair

@contextmanager
def temp_workspace():
    with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
        assert Path(folder).resolve().is_relative_to(ROOT.resolve())
        yield folder


class CPUEvent:
    def record(self): self.at=time.perf_counter()
    def elapsed_time(self,other): return (other.at-self.at)*1000


class ControlTests(unittest.TestCase):
    def test_timed_region_blocks_io_and_restores_after_failure(self):
        with temp_workspace() as folder:
            path=Path(folder)/'record.json'
            with self.assertRaisesRegex(RuntimeError,'inside measured'):
                with c.timed_region():
                    with c.timed_region(): c.write(path,{'value':1})
            self.assertFalse(path.exists())
            c.write(path,{'value':1},new=True)
            with self.assertRaises(FileExistsError): c.write(path,{'value':2},new=True)
            self.assertEqual(c.read(path),{'value':1})

    def test_inputs_and_balanced_order(self):
        from runner import inputs
        quality,bank=inputs()
        self.assertEqual(quality,[]); self.assertEqual(len(bank),8)
        self.assertEqual(c.CASES,[(8192,256,1),(32768,256,8)])
        orders=[c.arm_order(i,j) for i in (0,1) for j in (1,2,3)]
        self.assertEqual(sum(x[0]=='baseline' for x in orders),3)
        self.assertEqual(sum(x[0]=='optimized' for x in orders),3)

    def test_prepare_idempotence_and_changed_plan_rejection(self):
        import runner
        with temp_workspace() as folder:
            d=Path(folder); model=d/'model';model.mkdir();weight=model/'weight.bin';weight.write_bytes(b'weight')
            c.write(model/'model.safetensors.index.json',dict(weight_map={'x':'weight.bin'}))
            weights={'weight.bin':[weight.stat().st_size,weight.stat().st_mtime_ns]}
            cfg=dict(bench_manifest=c.BASE,bench=str(d/'bench'),model=str(model),weights=weights,old_out=str(d/'old'))
            c.write(d/'config.json',cfg);c.write(d/'old/results/phasekv/protocol.json',dict(model=str(model),weights=weights))
            args=SimpleNamespace(config=d/'config.json',out=d/'out');real_read=c.read
            def read(path):
                return {} if Path(path)==c.ROOT/'data/model_metadata.json' else real_read(path)
            with patch.object(c,'verify',return_value='manifest'),patch.object(c,'read',side_effect=read):
                a=runner.prepare(args);b=runner.prepare(args)
                self.assertEqual(a,b);self.assertEqual(a['quality'],[]);self.assertEqual(a['rounds'],3)
                c.write(args.out/'plan.json',dict(a,rounds=4))
                with self.assertRaises(ValueError):runner.prepare(args)

    def test_report_pairs_excludes_diagnostics_and_rejects_mismatch(self):
        with temp_workspace() as folder:
            out=Path(folder);c.write(out/'plan.json',dict(experiment='x'))
            c.write(out/'gates/session.json',dict(experiment='x',session='session',gpu_uuid='gpu',status='ok'))
            def pair(case,index,kind='formal'):
                arms={}
                for arm in c.ARMS:
                    row=dict(status='ok',arm=arm,gpu_uuid='gpu',input_tokens=case[0],batch=case[2],
                        output_tokens_per_row=case[1] if kind=='formal' else 1,effective_options=c.options(c.PROFILE),
                        input_sha256=['input'],output_sha256='output',first_token_sha256='first',packed_prompt_sha256='packed',
                        ttft_s=(index if arm=='baseline' else index*.8) if kind=='formal' else 999,
                        decode_s=10,aggregate_tokens_per_s=10,peak_allocated_bytes=index*2**30,
                        phase_timings=dict(stages={s:dict(cpu_call_ms=1,cuda_stream_ms=1) for s in (
                            'prefix_forward','observation_and_selection','selection','flatten','packing','first_token')}))
                    arms[arm]=row
                return dict(experiment='x',session='session',gpu_uuid='gpu',case=list(case),status='ok',kind=kind,
                            formal_round=index,order=list(c.ARMS),arms=arms)
            first=pair(c.CASES[0],1)
            c.write(out/f'pairs/{c.case_name(c.CASES[0])}/r1.json',first,new=True)
            partial=report(out);self.assertFalse(partial['complete']);self.assertEqual(partial['formal_pairs'],1)
            bad=json.loads(json.dumps(first));bad['arms']['optimized']['output_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'mismatch'):validate_pair(out,bad,'x',c.CASES[0],'formal',1)
            bad=json.loads(json.dumps(first));del bad['arms']['optimized']
            with self.assertRaisesRegex(ValueError,'Incomplete'):validate_pair(out,bad,'x',c.CASES[0],'formal',1)
            for case in c.CASES:
                for i in (1,2,3):
                    path=out/f'pairs/{c.case_name(case)}/r{i}.json'
                    if not path.exists():c.write(path,pair(case,i),new=True)
                c.write(out/f'diagnostics/{c.case_name(case)}.json',pair(case,1,'diagnostic'),new=True)
                c.write(out/f'warmups/{c.case_name(case)}.json',dict(ttft_s=100000))
            full=report(out);self.assertTrue(full['complete']);self.assertEqual(full['formal_pairs'],6)
            values=full['speed'][c.case_name(c.CASES[0])]
            self.assertEqual(values['baseline']['ttft_s'],2)
            self.assertEqual(values['baseline']['peak_gib'],3)
            self.assertAlmostEqual(values['median_paired_ttft_change_pct'],-20)


class InstrumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.set_num_threads(1);torch.manual_seed(411)
        cfg=LlamaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=1,max_position_embeddings=2048,vocab_size=128,attention_dropout=0.,_attn_implementation='eager')
        cls.model=LlamaForCausalLM(cfg).half().eval()
        class Wrapper(torch.nn.Module):
            def __init__(self,original):super().__init__();self.original=original
            def forward(self,*args,**kwargs):return self.original(*args,**kwargs)
        for layer in cls.model.model.layers:layer.self_attn=Wrapper(layer.self_attn)
        def forward(model,ids,cache,start,method,**kwargs):
            positions=torch.arange(start,start+ids.shape[1])
            result=model.model(input_ids=ids,past_key_values=cache,position_ids=positions[None],
                cache_position=positions,use_cache=True,return_dict=True,**kwargs)
            return model.lm_head(result.last_hidden_state[:,-1:,:]).float(),result.past_key_values,result.attentions
        cls.runtime=SimpleNamespace(forward=forward)

    def assert_cache_equal(self,a,b):
        import torch
        for x,y in zip(a.layers,b.layers):
            for name in ('key','value'):
                for field in ('packed','scale','mins'):
                    torch.testing.assert_close(getattr(getattr(x.prompt,name),field),getattr(getattr(y.prompt,name),field),rtol=0,atol=0)

    def test_real_llama_probe_preserves_logits_packing_and_nested_accounting(self):
        import torch
        from PhaseKV.memory_prefill import memory_prefill
        from PhaseKV.phasekv_cache import PhaseKVCache
        original=self.runtime.forward
        descriptor=PhaseKVCache.__dict__['from_dense_prefill']
        with torch.inference_mode():
            for batch,length in ((1,17),(2,612),(8,97)):
                ids=torch.randint(0,128,(batch,length)); outputs=[]
                for function in (baseline.speed_prefill,memory_prefill):
                    clean,cache=function(self.model,ids,c.options(c.PROFILE),self.runtime,{})
                    probe=StageProbe(self.runtime,length,32,CPUEvent,lambda:None)
                    with probe:
                        measured,other=function(self.model,ids,c.options(c.PROFILE),self.runtime,{})
                    torch.testing.assert_close(clean,measured,rtol=0,atol=0);self.assert_cache_equal(cache,other)
                    self.assertIs(self.runtime.forward,original)
                    self.assertIs(PhaseKVCache.__dict__['from_dense_prefill'],descriptor)
                    data=probe.result();s=data['stages']
                    self.assertEqual(s['observation_forward']['calls'],1)
                    self.assertEqual(s['packing']['calls'],1);self.assertEqual(s['flatten']['calls'],1)
                    self.assertEqual(s['selection']['calls'],1 if function==baseline.speed_prefill else 2)
                    if function==memory_prefill:
                        self.assertAlmostEqual(s['observation_and_selection']['cuda_stream_ms'],s['observation_forward']['cuda_stream_ms'])
                    else:
                        self.assertAlmostEqual(s['observation_and_selection']['cuda_stream_ms'],
                                               s['observation_forward']['cuda_stream_ms']+s['selection']['cuda_stream_ms'])
                    outputs.append((clean,cache))
                torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=0,atol=0)
                self.assert_cache_equal(outputs[0][1],outputs[1][1])

    def test_probe_restores_all_patches_on_exception(self):
        from PhaseKV.phasekv_cache import PhaseKVCache
        memory=importlib.import_module('PhaseKV.memory_prefill');prompt=importlib.import_module('PhaseKV.prompt_tier')
        original=(self.runtime.forward,baseline.flatten_requests,memory.flatten_owned_cache,
                  memory.select_prompt_indices,prompt.select_prompt_indices,PhaseKVCache.__dict__['from_dense_prefill'])
        with self.assertRaisesRegex(RuntimeError,'failure'):
            with StageProbe(self.runtime,97,32,CPUEvent,lambda:None):raise RuntimeError('failure')
        current=(self.runtime.forward,baseline.flatten_requests,memory.flatten_owned_cache,
                 memory.select_prompt_indices,prompt.select_prompt_indices,PhaseKVCache.__dict__['from_dense_prefill'])
        for a,b in zip(original,current):self.assertIs(a,b)


if __name__=='__main__':unittest.main(verbosity=2)
