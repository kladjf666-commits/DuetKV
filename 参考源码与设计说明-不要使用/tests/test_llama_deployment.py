"""Llama deployment checks. Tiny random models are not pretrained model benchmarks."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from models.registry import load_profile, validate_model_config, estimated_parameter_count
from models.Llama.integration import generate, build_input_ids
from models.Llama.packed_runtime import install_packed_attention
from models.packed_runtime import build_phasekv_cache, storage_evidence
from models.generation import stop_token_ids
from KIVI.packed_cache import PackedKIVICache
from PhaseKV.kernels.packing import unpack_dequantize

PROFILES = ('Llama-2-7b-chat-hf', 'Llama-3.1-8B-Instruct')
TOKENIZER = SimpleNamespace(eos_token_id=2, decode=lambda ids, **kw: str(ids))


def tiny_model(kv_heads, device='cpu', head_dim=32):
    cfg=LlamaConfig(vocab_size=113,hidden_size=4*head_dim,intermediate_size=96,
                    num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=kv_heads,
                    max_position_embeddings=512,attn_implementation='eager')
    if kv_heads == 1:
        cfg.rope_theta=500000.
        cfg.rope_scaling=load_profile(PROFILES[1])['architecture']['rope_scaling']
        cfg.max_position_embeddings=131072
    return LlamaForCausalLM(cfg).to(device=device,dtype=torch.bfloat16).eval()


class TestLlamaDeployment(unittest.TestCase):
    def test_profiles_parameter_count_and_real_architecture_guards(self):
        for name in PROFILES:
            with self.subTest(profile=name):
                p=load_profile(name)
                cfg=LlamaConfig(**p['architecture'])
                validate_model_config(p,cfg)
                self.assertEqual(len(p['compression']['methods']),6)
                with torch.device('meta'):
                    model=LlamaForCausalLM(cfg)
                count=sum(param.numel() for param in model.parameters())
                self.assertEqual(count,estimated_parameter_count(p['architecture']))
                self.assertEqual(count,6738415616 if name==PROFILES[0] else 8030261248)
                for key,value in [('rope_theta',123.),('pretraining_tp',2),('head_dim',64)]:
                    wrong=copy.deepcopy(cfg); setattr(wrong,key,value)
                    with self.assertRaisesRegex(ValueError,'architecture differs'):
                        validate_model_config(p,wrong)
                wrong=copy.deepcopy(cfg); wrong.rope_scaling={'rope_type':'linear','factor':2.}
                with self.assertRaisesRegex(ValueError,'rope_scaling'):
                    validate_model_config(p,wrong)

    @torch.inference_mode()
    def test_wrappers_are_transparent_and_dense_prefill_really_packs(self):
        for kv in (4,1):  # MHA as Llama-2-7B; 4:1 GQA as Llama-3.1-8B.
            with self.subTest(kv=kv):
                model=tiny_model(kv)
                ids=torch.arange(96).unsqueeze(0)
                before=model(input_ids=ids,use_cache=False).logits
                install_packed_attention(model); install_packed_attention(model)
                after=model(input_ids=ids,use_cache=False).logits
                self.assertTrue(torch.equal(before,after))
                prefill=model(input_ids=ids,past_key_values=DynamicCache(),use_cache=True,output_attentions=True)
                caches=[PackedKIVICache.from_dense_prefill(prefill.past_key_values),
                        build_phasekv_cache(prefill,model,96,obs_window=4)]
                for cache in caches:
                    evidence=storage_evidence(cache)
                    self.assertGreater(evidence['code_bytes'],0)
                    self.assertEqual(evidence['code_dtype'],'torch.uint8')
                    self.assertFalse(evidence['dense_shadow'])
                    self.assertLess(evidence['physical_bytes'],2*2*2*kv*96*32)

    @torch.inference_mode()
    def test_dense_methods_use_real_cache_and_trigger_eviction_on_llama(self):
        specs=[{'method':'full'}, {'method':'snapkv','max_cache_tokens':16,'obs_window':4},
               {'method':'scope','max_cache_tokens':4,'recent_window':2},
               {'method':'pyramidkv','max_cache_tokens':16,'obs_window':4}]
        for kv in (4,1):
            model=tiny_model(kv)
            install_packed_attention(model)  # Dense methods also work after KIVI/PhaseKV.
            for spec in specs:
                with self.subTest(kv=kv,method=spec['method']):
                    positions=[]
                    def record_positions(module,args,kwargs):
                        positions.append(kwargs['position_ids'].tolist())
                    hook=model.register_forward_pre_hook(record_positions,with_kwargs=True)
                    try:
                        out=generate(model,TOKENIZER,torch.arange(48).unsqueeze(0),spec,
                                     max_new_tokens=10,stop_on_eos=False)
                    finally:
                        hook.remove()
                    self.assertEqual(len(out['generated_ids']),10)
                    self.assertEqual(positions[1:],[[[i]] for i in range(48,57)])
                    trace=out['compression_trace']
                    if spec['method']=='full':
                        self.assertEqual(trace['final_state']['physical_bytes'],2*2*2*kv*57*32)
                    else:
                        self.assertEqual(trace['status'],'triggered')

    def test_packed_methods_fail_closed_and_receive_multi_eos(self):
        model=SimpleNamespace(config=SimpleNamespace(model_type='llama',max_position_embeddings=512),
                              generation_config=SimpleNamespace(eos_token_id=[2,3,4]))
        self.assertEqual(stop_token_ids(model,TOKENIZER),{2,3,4})
        with patch('models.Llama.integration.generate_packed',return_value={
                'storage_evidence':{'dense_shadow':True,'code_dtype':'torch.uint8'}}):
            for method in ('kivi','phasekv'):
                with self.assertRaisesRegex(AssertionError,'genuine packed'):
                    generate(model,TOKENIZER,torch.ones((1,16),dtype=torch.long),{'method':method},max_new_tokens=4)
        with patch('models.Llama.packed_runtime._generate_packed',return_value={}) as runner:
            from models.Llama.packed_runtime import generate_packed
            generate_packed(model,TOKENIZER,torch.ones((1,16)),method='kivi',params={},max_new_tokens=4)
            self.assertEqual(runner.call_args.kwargs['eos_token_ids'],{2,3,4})

    @torch.inference_mode()
    def test_llama_dense_decode_honors_all_eos_ids_and_fixed_output(self):
        model=tiny_model(1)
        ids=torch.tensor([[7,8,9]])
        token=int(model(input_ids=ids).logits[0,-1].argmax())
        # Chosen token is a second generation EOS, different from tokenizer EOS.
        tokenizer=SimpleNamespace(eos_token_id=(token+1)%113,decode=TOKENIZER.decode)
        model.generation_config.eos_token_id=[(token+2)%113,token]
        out=generate(model,tokenizer,ids,{'method':'full'},max_new_tokens=5)
        self.assertEqual(out['generated_ids'],[token])
        out=generate(model,tokenizer,ids,{'method':'full'},max_new_tokens=5,stop_on_eos=False)
        self.assertEqual(len(out['generated_ids']),5)

    def test_template_adds_bos_once_and_uses_body_token_budget(self):
        tokenizer=Mock()
        tokenizer.chat_template='native'
        tokenizer.return_value=SimpleNamespace(input_ids=list(range(20)))
        tokenizer.decode.side_effect=lambda ids,**kw: ','.join(map(str,ids))
        tokenizer.apply_chat_template.return_value=torch.tensor([[1,7,8]])
        out=build_input_ids(tokenizer,'hotpotqa',{'context':'context','input':'question'},7)
        self.assertEqual(out.tolist(),[[1,7,8]])
        self.assertFalse(tokenizer.call_args.kwargs['add_special_tokens'])
        self.assertTrue(tokenizer.apply_chat_template.call_args.kwargs['tokenize'])
        self.assertEqual(tokenizer.decode.call_args_list[0].args[0],[0,1,2])
        self.assertEqual(tokenizer.decode.call_args_list[1].args[0],[16,17,18,19])


class TestLlamaGPU(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'Requires rented CUDA GPU; no pretrained weights needed')
    @torch.inference_mode()
    def test_packed_decode_matches_materialized_reference_for_mha_and_gqa(self):
        """Dequantization is reference-only here; production attention reads packed codes."""
        for kv in (4,1):
            for method in ('kivi','phasekv'):
                with self.subTest(kv=kv,method=method):
                    torch.manual_seed(12)
                    model=tiny_model(kv,device='cuda',head_dim=128)
                    reference=copy.deepcopy(model)
                    ids=torch.arange(63,device='cuda').unsqueeze(0)
                    prefill=model(input_ids=ids,past_key_values=DynamicCache(),use_cache=True)
                    if method=='kivi':
                        cache=PackedKIVICache.from_dense_prefill(prefill.past_key_values)
                    else:
                        cache=build_phasekv_cache(prefill,model,63,prefill_keep_ratio=1.,recent_window=4,lazy_g=2)
                    del prefill
                    install_packed_attention(model)
                    for step in range(8):
                        dc=DynamicCache()
                        for idx,layer in enumerate(cache.layers):
                            if method=='kivi':
                                def joined(p,res):
                                    return torch.cat([unpack_dequantize(p),res],dim=2) if p else res
                                k,v=joined(layer.key,layer.key_residual),joined(layer.value,layer.value_residual)
                            else:
                                segments=[layer.prompt,*layer.prompt_extensions,*layer.cold]
                                k=torch.cat([unpack_dequantize(s.key) for s in segments]+[layer.hot_views()[0]],dim=2)
                                v=torch.cat([unpack_dequantize(s.value) for s in segments]+[layer.hot_views()[1]],dim=2)
                            dc.update(k.bfloat16(),v.bfloat16(),idx)
                        positions=torch.tensor([[63+step]],device='cuda')
                        kwargs=dict(input_ids=torch.tensor([[64+step]],device='cuda'),use_cache=True,
                                    position_ids=positions,cache_position=torch.tensor([cache.get_seq_length()],device='cuda'))
                        expected=reference(past_key_values=dc,**kwargs).logits
                        actual=model(past_key_values=cache,**kwargs).logits
                        cache.finish_decode_step()
                        torch.testing.assert_close(actual,expected,atol=0.02,rtol=0.03)
                        storage_evidence(cache)
                    del model,reference,cache


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
