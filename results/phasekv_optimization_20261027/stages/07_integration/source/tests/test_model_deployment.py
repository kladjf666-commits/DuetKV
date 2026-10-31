"""Deployment routing/configuration checks without loading pretrained 7B weights."""
import copy
import importlib
import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from models.registry import load_profile, validate_methods, validate_model_config, estimated_parameter_count
from models.Qwen2.integration import generate
from eval.compare_models import load_named_model, preflight, main


class TestModelDeployment(unittest.TestCase):
    def test_profiles_route_all_six_methods_and_7b_gqa_dimensions(self):
        for name in ('Qwen2-7B-Instruct','Qwen2.5-7B-Instruct','Qwen2-1.5B-Instruct'):
            profile=load_profile(name)
            self.assertEqual([s['method'] for s in profile['compression']['methods']],
                             ['full','kivi','snapkv','scope','pyramidkv','phasekv'])
            cfg=Qwen2Config(**profile['architecture'])
            validate_model_config(profile,cfg)
            if '7B' in name:
                self.assertEqual((cfg.num_attention_heads,cfg.num_key_value_heads,cfg.hidden_size//cfg.num_attention_heads),(28,4,128))

    def test_parameter_count_matches_meta_model_without_allocating_weights(self):
        profile=load_profile('Qwen2-7B-Instruct')
        with torch.device('meta'):
            model=Qwen2ForCausalLM(Qwen2Config(**profile['architecture']))
        actual=sum(p.numel() for p in model.parameters())
        self.assertEqual(estimated_parameter_count(profile['architecture']),actual)
        self.assertGreater(actual*2,8*2**30)

    def test_wrong_model_rejected_before_weight_loading(self):
        profile=load_profile('Qwen2-7B-Instruct')
        wrong=Qwen2Config(**load_profile('Qwen2-1.5B-Instruct')['architecture'])
        with patch('eval.compare_models.AutoConfig.from_pretrained',return_value=wrong), \
             patch('eval.compare_models.AutoModelForCausalLM.from_pretrained') as loader:
            with self.assertRaisesRegex(ValueError,'architecture differs'):
                load_named_model(profile,SimpleNamespace(model_path='/wrong-model',allow_download=False))
            loader.assert_not_called()
        right=Qwen2Config(**profile['architecture'])
        right.quantization_config={'load_in_4bit':True}
        with self.assertRaisesRegex(ValueError,'Weight-quantized'):
            validate_model_config(profile,right)

    def test_invalid_method_configs_fail_before_model_loading(self):
        profile=load_profile('Qwen2-7B-Instruct')
        invalid=[ [{'method':'kvquant'}], [{'method':'full','bits':4}],
                  [{'method':'kivi','bits':3}], [{'method':'kivi','group_size':30}],
                  [{'method':'scope','recent_window':16,'max_cache_tokens':32,'decoding_metric':'adaptive'}],
                  [{'method':'full'},{'method':'full'}],
                  [{'method':'phasekv','prompt_bits':16}],
                  [{'method':'phasekv','fake_quant':True}]]
        for methods in invalid:
            with self.subTest(methods=methods), self.assertRaises(ValueError):
                validate_methods(profile,methods)

    def test_method_dispatch_keeps_quantizers_on_packed_path(self):
        profile=load_profile('Qwen2-7B-Instruct')
        model=SimpleNamespace(config=SimpleNamespace(model_type='qwen2',max_position_embeddings=32768))
        ids=torch.zeros((1,96),dtype=torch.long)
        packed={'storage_evidence':{'dense_shadow':False,'code_dtype':'torch.uint8'}}
        with patch('models.Qwen2.integration.generate_packed',return_value=packed) as pack, \
             patch('models.Qwen2.integration.runtime.generate_with_compression',return_value={'ok':True}) as dense:
            for spec in profile['compression']['methods']:
                generate(model,None,ids,spec,max_new_tokens=80,stop_on_eos=False)
            self.assertEqual(pack.call_count,2)
            self.assertEqual({c.kwargs['method'] for c in pack.call_args_list},{'kivi','phasekv'})
            self.assertEqual(dense.call_count,4)
            self.assertTrue(all(c.kwargs['trace_compression'] for c in dense.call_args_list))
        with self.assertRaisesRegex(ValueError,'context length'):
            generate(model,None,ids,{'method':'full'},max_new_tokens=32768)

    def test_preflight_rejects_8gb_for_full_bf16_7b(self):
        profile=load_profile('Qwen2-7B-Instruct')
        with patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.mem_get_info',return_value=(7*2**30,8*2**30)), \
             patch('torch.cuda.is_bf16_supported',return_value=True), \
             patch('torch.cuda.get_device_name',return_value='test 8GB'), \
             patch('eval.compare_models.checkpoint_status',return_value={'weights_present':True}):
            result=preflight(profile,profile['compression']['methods'])
        self.assertFalse(result['ready'])
        self.assertTrue(any('VRAM' in s for s in result['blocking_reasons']))

    def test_moved_modules_keep_identity_for_existing_callers(self):
        for a,b in [('qwen2.qwen2','models.Qwen2.qwen2'),
                    ('PhaseKV.qwen2.qwen2_attention','models.Qwen2.qwen2_attention'),
                    ('KIVI.qwen2_attention','models.Qwen2.kivi_attention'),
                    ('eval.packed_runtime','models.Qwen2.packed_runtime'),
                    ('llama.llama','models.Llama.llama'),('opt.opt','models.OPT.opt')]:
            self.assertIs(importlib.import_module(a),importlib.import_module(b))

    def test_comparison_reports_shared_inputs_and_keeps_existing_results(self):
        """Exercise the real reporting loop with mocked GPU inference, not a model score."""
        profile=load_profile('Qwen2-7B-Instruct')
        cfg=Qwen2Config(**profile['architecture'])
        ids=torch.tensor([[1,2,3]])
        samples=[{'sample_id':f'example:{i}','task':'example','ids':ids.clone(),
                  'answers':['answer'],'max_new':2} for i in range(2)]
        full_bytes=2*2*cfg.num_hidden_layers*cfg.num_key_value_heads*128*4
        trace={'status':'not_triggered','summary':{'decode':{'triggered_steps':0}},
               'final_state':{'physical_bytes':full_bytes}}
        calls=[]
        def fake_generate(model,tokenizer,input_ids,spec,**kwargs):
            calls.append((spec['method'],input_ids.tolist(),kwargs))
            return {'prompt_len':3,'generated_ids':[4,5],'text':'answer',
                    'peak_gpu_bytes':1234,'compression_trace':copy.deepcopy(trace)}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            prefix=Path(directory)/'report'
            stack.enter_context(patch('eval.compare_models.preflight',return_value={'ready':True}))
            loader=stack.enter_context(patch('eval.compare_models.load_named_model',
                return_value=(SimpleNamespace(config=cfg),None)))
            stack.enter_context(patch('eval.compare_models.prepare_samples',return_value=(samples,{'example':'test-hash'})))
            stack.enter_context(patch('eval.compare_models.integration',return_value=SimpleNamespace(generate=fake_generate)))
            stack.enter_context(patch('eval.compare_models.forbid_fake_quantization'))
            for name in ('synchronize','empty_cache'):
                stack.enter_context(patch(f'torch.cuda.{name}'))
            stack.enter_context(patch('torch.cuda.get_device_name',return_value='mocked GPU'))
            args=['--profile',profile['name'],'--mode','longbench','--out',str(prefix)]
            self.assertEqual(main(args),0)
            report=json.loads(prefix.with_suffix('.json').read_text())
            records=[json.loads(line) for line in prefix.with_suffix('.predictions.jsonl').read_text().splitlines()]
            self.assertEqual(report['meta']['status'],'complete')
            self.assertEqual(len(report['rows']),6)
            self.assertEqual(len(records),12)
            self.assertEqual(len(calls),18)  # One warmup plus two samples per method.
            self.assertTrue(all(call[1]==ids.tolist() for call in calls))
            self.assertEqual(len({r['input_hash'] for r in records}),1)
            self.assertTrue(all(r['qa_f1']==100 for r in report['rows']))
            self.assertTrue(all(r['exact_matches_vs_full']==2 for r in report['rows']))
            self.assertIn('source_hashes',report['meta'])
            self.assertTrue(prefix.with_suffix('.md').is_file())
            with self.assertRaises(FileExistsError):
                main(args)
            loader.assert_called_once()


if __name__=='__main__': unittest.main()
