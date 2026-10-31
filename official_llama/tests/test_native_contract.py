"""CPU API/semantics tests. Test-only flash shim is NEVER a runtime fallback."""
from pathlib import Path
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare
import native_runtime as runtime
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache


def flash_reference(q, k, v, dropout_p=0., softmax_scale=None, causal=True, **kwargs):
    # Bottom-right causal alignment, including q_len=1 decode, mirrors FA >=2.1.
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    if q.shape[1] != k.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    mask = None
    if causal:
        rows, cols = q.shape[-2], k.shape[-2]
        mask = torch.arange(cols)[None] <= torch.arange(cols - rows, cols)[:, None]
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                dropout_p=0., scale=softmax_scale).transpose(1, 2)


def forbidden(*args, **kwargs):
    raise AssertionError('This CPU test must not call a CUDA quantization kernel')


def shim_modules():
    flash = types.ModuleType('flash_attn')
    flash.flash_attn_func = flash_reference
    flash.flash_attn_varlen_func = forbidden
    padding = types.ModuleType('flash_attn.bert_padding')
    padding.index_first_axis = padding.pad_input = padding.unpad_input = forbidden
    packing = types.ModuleType('quant.new_pack')
    packing.triton_quantize_and_pack_along_last_dim = forbidden
    matmul = types.ModuleType('quant.matmul')
    matmul.cuda_bmm_fA_qB_outer = forbidden
    return {'flash_attn': flash, 'flash_attn.bert_padding': padding,
            'quant.new_pack': packing, 'quant.matmul': matmul}


def config():
    return LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=131072, rope_theta=500000.,
                        rope_scaling={'rope_type': 'llama3', 'factor': 8.,
                                      'low_freq_factor': 1., 'high_freq_factor': 4.,
                                      'original_max_position_embeddings': 8192})


class NativeContract(unittest.TestCase):
    def test_pristine_sources_and_unmodified_kernels(self):
        manifest = prepare.verify()
        for name in prepare.REPOS:
            for file in (ROOT / 'vendor' / name).rglob('*.py'):
                if '.git' in file.parts:
                    continue
                relative = file.relative_to(ROOT / 'vendor')
                if relative.as_posix() in ('SnapKV/snapkv/monkeypatch/llama_hijack_4_37.py',
                                          'SCOPE/model/llama_model.py', 'KIVI/models/llama_kivi.py'):
                    continue
                self.assertEqual(file.read_bytes(), (ROOT / 'runtime' / relative).read_bytes())
        self.assertEqual(len(manifest['repositories']), 3)

    def test_scope_boundary_and_delta(self):
        options = json.loads((ROOT / 'config.json').read_text())['methods']['scope']
        for limit in (32, 128, 256, 511):
            active, mode = runtime.scope_schedule(options, limit)
            self.assertEqual(active['decoding_metric'], 'None')
            self.assertIn('boundary', mode)
        active, _ = runtime.scope_schedule(options, 4096)
        self.assertEqual(active['delta'], 15)
        self.assertEqual(active['decoding_metric'], 'discontinuous')

    def test_storage_counts_backing_allocation_once(self):
        backing = torch.zeros((1, 2, 16, 8), dtype=torch.float16)
        self.assertEqual(runtime.cache_report(((backing[:, :, :2], backing),))['physical_bytes'],
                         backing.untyped_storage().nbytes())

    def test_kivi_native_class_and_uncompressed_decode(self):
        cfg = config()
        cfg.k_bits = cfg.v_bits = 4
        cfg.group_size = 32
        cfg.residual_length = 128
        cfg.use_flash = cfg._flash_attn_2_enabled = True
        with patch.dict(sys.modules, shim_modules()):
            native = runtime.module_at('test_native_kivi', ROOT / 'runtime/KIVI/models/llama_kivi.py')
            torch.manual_seed(4)
            actual = native.LlamaForCausalLM_KIVI(cfg).eval()
            expected = LlamaForCausalLM(config()).eval()
            expected.load_state_dict(actual.state_dict(), strict=True)
            ids = torch.arange(16)[None]
            with torch.no_grad():
                got = actual(ids, use_cache=True)
                ref = expected(ids, use_cache=True)
                torch.testing.assert_close(got.logits, ref.logits, atol=2e-5, rtol=2e-4)
                got_next = actual(torch.tensor([[17]]), past_key_values=got.past_key_values, use_cache=True)
                ref_next = expected(torch.tensor([[17]]), past_key_values=ref.past_key_values, use_cache=True)
                torch.testing.assert_close(got_next.logits, ref_next.logits, atol=2e-5, rtol=2e-4)

    def test_snap_scope_native_prefill_and_compacted_decode(self):
        # Real native attention+selector; only GPU FA2 is substituted on CPU.
        for method in ('snapkv', 'scope'):
            with self.subTest(method=method), patch.dict(sys.modules, shim_modules()):
                cfg = config()
                model = LlamaForCausalLM(cfg).eval()
                options = {'max_capacity_prompt': 24, 'window_size': 8,
                           'kernel_size': 7, 'pooling': 'maxpool'}
                if method == 'scope':
                    options.update(decoding_window_size=16, decoding_recent_size=8,
                                   decoding_metric='discontinuous')
                original = model.model.layers[0].self_attn
                original._flash_attn_uses_top_left_mask = False
                original.is_causal = True
                runtime.install(model, method, options)
                runtime.reset_sample(model, method, options, 64)
                hidden = torch.randn(1, 40, 64)
                positions = torch.arange(40)[None]
                cache = DynamicCache()
                with torch.no_grad():
                    out, _, _ = original(hidden, position_ids=positions, past_key_value=cache, use_cache=True)
                    self.assertEqual(out.shape, hidden.shape)
                    self.assertEqual(cache.key_cache[0].shape, (1, 4, 24, 16))
                    # Compact KV must retain native repeated query heads (4, not 2).
                    dense_k, dense_v = cache.key_cache[0].clone(), cache.value_cache[0].clone()
                    h = torch.randn(1, 1, 64)
                    pos = torch.tensor([[40]])
                    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
                    q = original.q_proj(h).view(1, 1, 4, 16).transpose(1, 2)
                    k = original.k_proj(h).view(1, 1, 2, 16).transpose(1, 2)
                    v = original.v_proj(h).view(1, 1, 2, 16).transpose(1, 2)
                    cos, sin = original.rotary_emb(v, pos)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    k = torch.cat((dense_k, repeat_kv(k, 2)), dim=2)
                    v = torch.cat((dense_v, repeat_kv(v, 2)), dim=2)
                    reference = flash_reference(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                    reference = original.o_proj(reference.reshape(1, 1, 64))
                    got, _, _ = original(h, position_ids=pos, past_key_value=cache, use_cache=True)
                    torch.testing.assert_close(got, reference)

    def test_phase_observation_chunk_preserves_dense_prefill(self):
        from PhaseKV.phasekv_cache import PhaseKVCache
        model = LlamaForCausalLM(config()).eval()
        options = json.loads((ROOT / 'config.json').read_text())['methods']['phasekv']
        runtime.install(model, 'phasekv', options)
        ids = torch.arange(64)[None]
        captured = {}

        def capture(dense, **kwargs):
            captured.update(kwargs)
            return dense

        with torch.no_grad(), patch.object(PhaseKVCache, 'from_dense_prefill', side_effect=capture):
            expected, _, _ = runtime.forward(model, ids, DynamicCache(), 0, 'phasekv')
            actual, cache = runtime.phase_prefill(model, ids, options)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        self.assertEqual(cache.get_seq_length(), 64)
        self.assertEqual(captured['keep_indices'][0].shape, (1, 2, round(64 * .7)))

    def test_scope_default_smoke_actually_evicts_decode(self):
        sys.path.insert(0, str(ROOT / 'runtime/SCOPE'))
        from model.kv_utils import H2OKVCluster
        base = json.loads((ROOT / 'config.json').read_text())['methods']['scope']
        for mode in ('slide', 'adaptive', 'discontinuous'):
            options, _ = runtime.scope_schedule({**base, 'decoding_metric': mode}, 1024)
            cluster = H2OKVCluster(num_hidden_layers=1, **options)
            H2OKVCluster.current_decoding_step = H2OKVCluster.jump_step = H2OKVCluster.jump_layer = 0
            prefix = torch.randn(1, 1, 2048, 4)
            k, v = prefix.clone(), prefix.clone()
            for step in range(1023):
                q = torch.randn(1, 1, 1, 4)
                k = torch.cat((k, q), dim=2)
                v = torch.cat((v, q), dim=2)
                k, v = cluster.update_kv_in_decoding(k, q, v, None, 1)
            self.assertLess(k.shape[2], 2048 + 1023, mode)
            torch.testing.assert_close(k[:, :, :2048], prefix)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
