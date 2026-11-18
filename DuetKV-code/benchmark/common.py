"""A800 Prompt90 experiment configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = dict(id='J_p90_w128_o32', prefill_keep_ratio=.9, decode_keep_ratio=.8,
               prompt_bits=4, cold_bits=4, recent_window=128, obs_window=32)
OPT = dict(fused_quantize=True, reuse_workspace=True, paged_cold=True, page_slots=32,
           cold_group=8, specialized_4bit=True, gqa_reuse=True, hot_ring=True,
           channel_contiguous_k=True, a800_prompt_mma=True)

def options(p):
    return dict({k: v for k, v in p.items() if k != 'id'}, group_size=32,
                lazy_g=16, sink_tokens=0, optimizations=dict(OPT))
