"""CPU regression checks for the new runner, independent of GPU availability."""
from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import benchmark
import benchmark_metrics
import benchmark_generate


class BenchmarkTests(unittest.TestCase):
    def test_metrics_missing_answers(self):
        s={'suite':'longgenbench','task':'mmlu','answers':['A','B','C'],'first_answer_number':6}
        value=benchmark_metrics.score(s,'Answer_6: The answer is (A)\nAnswer_8: The answer is (C)')
        self.assertAlmostEqual(value['score'],100/3)
        self.assertAlmostEqual(value['indexed_accuracy_diagnostic'],200/3)
        self.assertEqual(value['unanswered_questions'],1)
        self.assertEqual(value['last_third_indexed_accuracy'],100)

    def test_official_metrics(self):
        self.assertEqual(benchmark_metrics.score({'suite':'longbench','task':'qasper','answers':['the red fox']},'red fox')['score'],100)
        r={'suite':'ruler','task':'niah_multivalue','ruler_base_task':'niah','answers':['123','456']}
        self.assertEqual(benchmark_metrics.score(r,'123')['score'],50)
        self.assertEqual(benchmark_metrics.score(r,'')['score'],0)

    def test_resume_and_failure_denominators(self):
        s={'id':'longbench/qasper/0/8192','suite':'longbench','task':'qasper','methods':['full'],
           'length_budget':8192,'input_sha256':'a','cluster_id':'qasper/0'}
        with tempfile.TemporaryDirectory() as tmp:
            p=benchmark.result_path(tmp,'full',s,0)
            self.assertTrue(benchmark.pending(p))
            benchmark.atomic(p,{**s,'method':'full','repeat':0,'status':'oom'})
            self.assertFalse(benchmark.pending(p))
            self.assertTrue(benchmark.pending(p,True))
            status=benchmark.summarize(tmp,[s],['longbench'],['full'])
            self.assertEqual(status['oom'],1)
            summary=json.loads((Path(tmp)/'summary.json').read_text())
            self.assertFalse(summary['all_succeeded'])
            self.assertTrue(summary['completed'])
            self.assertEqual(summary['groups'][0]['expected'],1)

    def test_generation_counts_eos_and_fixed_output(self):
        import torch
        fake=types.ModuleType('native_runtime')
        class Cache:
            n=0
            def finish_decode_step(self): pass
        def fwd(model,ids,cache,start,method):
            cache=cache or Cache(); cache.n+=ids.shape[1]
            logits=torch.zeros((1,1,8)); logits[0,0,2]=1
            return logits,cache,None
        fake.reset_sample=lambda *a: ({},'native')
        fake.forward=fwd
        fake.phase_prefill=lambda model,ids,opts:fwd(model,ids,None,0,'phasekv')[:2]
        fake.cache_report=lambda cache:{'physical_bytes':cache.n*16,'auxiliary_bytes':0}
        fake.cache_trace=lambda cache,method:{'physical_tokens':cache.n}
        model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=1000))
        original_tensor=torch.tensor
        def cpu_tensor(*a,**kw):
            kw.pop('device',None); return original_tensor(*a,**kw)
        s={'id':'x','input_ids':[1,3,4],'max_new_tokens':7,'stop_ids':[2],'natural_eos':True}
        with patch.dict(sys.modules,{'native_runtime':fake}), patch('torch.tensor',cpu_tensor), \
             patch('torch.cuda.synchronize'),patch('torch.cuda.reset_peak_memory_stats'), \
             patch('torch.cuda.max_memory_allocated',return_value=0),patch('torch.cuda.max_memory_reserved',return_value=0):
            r=benchmark_generate.generate(model,'full',{},s)
            self.assertEqual(r['output_tokens'],1)
            self.assertEqual(r['decode_s'],0)
            self.assertTrue(r['stopped_on_eos'])
            s['natural_eos']=False
            r=benchmark_generate.generate(model,'phasekv',{},s,diagnostic=True)
            self.assertEqual(r['output_tokens'],7)
            self.assertEqual(sum(c['tokens'] for c in r['decode_chunks']),6)
            self.assertEqual(r['final_storage']['physical_bytes'],9*16)
            self.assertEqual(r['persistent_kv_peak_bytes_diagnostic'],9*16)
            self.assertTrue(r['diagnostic_pass'])


if __name__=='__main__':
    unittest.main()
