"""Fixed-length decoding with the timing and memory counters used on A800."""
import gc
import time

import torch
from DuetKV import runtime as rt
from DuetKV.batched_attention import BatchedDuetKVAttention
from DuetKV.memory_prefill import memory_prefill


class DuetKVEngine:
    def __init__(self, model_dir, options):
        self.model, _, _ = rt.load_model(model_dir, 'duetkv')
        for layer in self.model.model.layers:
            layer.self_attn = BatchedDuetKVAttention(layer.self_attn.original)
        self.options = options

    def start(self, ids, limit, quality, progress):
        self.offset = ids.shape[1]
        self.effective, self.mode = rt.reset_sample(self.model, 'duetkv', self.options, limit)
        logits, self.cache = memory_prefill(self.model, ids, self.options, rt, progress,
                                           prefix_chunk_size=None if quality else 512)
        return logits

    def step(self, token):
        logits, self.cache, _ = rt.forward(self.model, token, self.cache, self.offset, 'duetkv')
        self.cache.finish_decode_step()
        self.offset += 1
        return logits

    def release(self):
        self.cache = None
        gc.collect()
        torch.cuda.empty_cache()


def run(e,rows,limit,stops,quality,progress):
    ids=torch.tensor(rows,dtype=torch.long,device='cuda')
    if ids.ndim!=2 or limit<1:raise ValueError('Bad shape')
    if quality and ids.shape[0]!=1:raise ValueError('Quality uses B1 natural EOS')
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    logits=e.start(ids,limit,quality,progress);token=logits[:,-1].argmax(-1,keepdim=True)
    torch.cuda.synchronize();ttft=time.perf_counter()-start;prefill_peak=torch.cuda.max_memory_allocated()
    if not torch.isfinite(logits).all(): raise ValueError('nonfinite prefill')
    tokens=[token];progress['stage']='decode'
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    while len(tokens)<limit:
        if stops and int(token.item()) in stops:break
        progress['step']=len(tokens)
        logits=e.step(token);token=logits[:,-1].argmax(-1,keepdim=True);tokens.append(token)
    torch.cuda.synchronize();sec=time.perf_counter()-start;dp=torch.cuda.max_memory_allocated()
    if not torch.isfinite(logits).all():raise ValueError('nonfinite decode')
    indices=torch.cat(tokens,dim=1).tolist();steps=len(tokens)-1
    return dict(batch=len(rows),input_tokens=len(rows[0]),output_tokens_per_row=len(tokens),decode_steps=steps,
        decoded_tokens=steps*len(rows),decode_s=sec,aggregate_tokens_per_s=steps*len(rows)/sec if steps else None,
        per_sequence_tokens_per_s=steps/sec if steps else None,ttft_s=ttft,prefill_peak_bytes=prefill_peak,
        decode_peak_bytes=dp,peak_allocated_bytes=max(dp,prefill_peak),output_ids=indices,
        effective_options=getattr(e,'effective',e.options),mode=getattr(e,'mode','native'))
