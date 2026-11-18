"""Private worker: imports the existing frozen runtime; changes only profile options."""
from pathlib import Path
from types import SimpleNamespace
import argparse,gc,os,platform,signal,sys,time,traceback
import common as c

def make_engine_class(NativeEngine,rt):
    class ProfileEngine(NativeEngine):
        def configure(self,p):
            self.release();self.options=c.options(p);self.profile=p
        def start(self,ids,limit,quality,progress):
            if quality and ids.shape[0]!=1:raise ValueError('Quality requires B1 natural EOS')
            self.offset=ids.shape[1];progress['stage']='prefill'
            self.effective,self.mode=rt.reset_sample(self.model,'phasekv',self.options,limit)
            if quality:logits,self.cache=rt.phase_prefill(self.model,ids,self.options)
            else:logits,self.cache=speed_prefill(self.model,ids,self.options,rt,progress)
            return logits
        def evidence(self):
            value=super().evidence();cache=self.cache
            value['trace'].update(prompt_bits=cache.prompt_bits,cold_bits=cache.cold_bits,
                recent_window=cache.recent_window,lazy_g=cache.lazy_g,decode_keep_ratio=cache.decode_keep_ratio,
                hot_tokens_by_layer=[x.hot_len for x in cache.layers],
                cold_tokens_by_layer=[sum(s.length for s in x.cold) for x in cache.layers])
            value['trace']['packed_heads_by_layer']=[x.hot_k.shape[1] for x in cache.layers]
            return value
    return ProfileEngine

def check_measurement(value,p,length):
    if value['effective_options']!=c.options(p):raise ValueError('Effective settings differ from requested profile')
    trace=value['trace'];wanted=round(length*p['prefill_keep_ratio'])
    if not all(x==wanted for x in trace['prompt_tokens_by_layer']):raise ValueError('Actual prompt retention differs')
    if trace['absolute_next_position']!=length+value['decode_steps']:raise ValueError('Absolute positions changed by prompt pruning')
    if (trace['prompt_bits']!=p['prompt_bits'] or trace['cold_bits']!=p['cold_bits'] or
        trace['recent_window']!=p['recent_window'] or trace['decode_keep_ratio']!=p['decode_keep_ratio']):raise ValueError('Actual cache settings differ')
    steps=value['decode_steps'];cycles=max(0,(steps-p['recent_window'])//16)
    hot=steps-cycles*16;cold=cycles*max(1,round(16*p['decode_keep_ratio']))
    if not all(x==hot for x in trace['hot_tokens_by_layer']) or not all(x==cold for x in trace['cold_tokens_by_layer']):raise ValueError('Actual generated-token retention differs')
    if len(trace['prompt_tokens_by_layer'])!=32:raise ValueError('Expected all 32 model layers')
    if trace.get('packed_heads_by_layer')!=[8*value['batch']]*32:raise ValueError('Batch/head isolation geometry differs')


def flatten_requests(dense,keep):
    """Map request b/head h to head b*H+h, preserving the existing batched decode contract."""
    from types import SimpleNamespace
    keys=[];values=[];indices=[]
    for k,v,ix in zip(dense.key_cache,dense.value_cache,keep):
        b,h,n,d=k.shape
        if v.shape!=k.shape or ix.shape[:2]!=(b,h):raise ValueError('Invalid selected request geometry')
        keys.append(k.reshape(1,b*h,n,d));values.append(v.reshape(1,b*h,n,d))
        indices.append(ix.reshape(1,b*h,ix.shape[-1]))
    return SimpleNamespace(key_cache=keys,value_cache=values),indices


def speed_prefill(model,ids,options,rt,progress):
    """Batched native FA2 prefix, eager observation tail, per-request selection, packed heads.

    Chunking bounds speed prefill memory as in the original speed runner. All work
    stays inside Engine.start and therefore inside measured TTFT. No request loop.
    """
    import torch
    from types import MethodType
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import LlamaAttention
    from PhaseKV.phasekv_cache import PhaseKVCache
    from PhaseKV.prompt_tier import select_prompt_indices
    length=ids.shape[1];observation=min(options['obs_window'],length);start=length-observation
    cache=DynamicCache()
    for i in range(0,start,512):
        progress['prefill_tokens']=min(i+512,start)
        _,cache,_=rt.forward(model,ids[:,i:min(i+512,start)],cache,i,'phasekv')
    originals=[layer.self_attn.original for layer in model.model.layers]
    saved=[attention.forward for attention in originals]
    mask=torch.zeros((1,1,observation,length),device=ids.device,dtype=torch.float16)
    future=torch.arange(length,device=ids.device)[None]>torch.arange(start,length,device=ids.device)[:,None]
    mask.masked_fill_(future[None,None],torch.finfo(torch.float16).min)
    try:
        for attention in originals:attention.forward=MethodType(LlamaAttention.forward,attention)
        logits,dense,attentions=rt.forward(model,ids[:,start:],cache,start,'phasekv',attention_mask=mask,output_attentions=True)
    finally:
        for attention,previous in zip(originals,saved):attention.forward=previous
    progress['stage']='select_and_pack_prompt'
    keep=select_prompt_indices(attentions,prompt_len=length,model_config=model.config,
        prefill_keep_ratio=options['prefill_keep_ratio'],obs_window=options['obs_window'],sink_tokens=options['sink_tokens'],pool_kernel=7,pooling='maxpool')
    dense,keep=flatten_requests(dense,keep)
    packing={k:options[k] for k in ('prompt_bits','cold_bits','group_size','recent_window','lazy_g','decode_keep_ratio','optimizations')}
    return logits,PhaseKVCache.from_dense_prefill(dense,prompt_len=length,keep_indices=keep,**packing)

def synthetic_gate(options,batch=1):
    import torch
    from PhaseKV.phasekv_cache import PhaseKVCache
    from PhaseKV.kernels.segmented_attention import segmented_phasekv_decode
    from PhaseKV.kernels.packing import unpack_dequantize
    rng=torch.Generator(device='cuda').manual_seed(31415)
    rand=lambda *shape:torch.randn(*shape,device='cuda',dtype=torch.float16,generator=rng)*.2
    heads=8*batch
    dense=SimpleNamespace(key_cache=[rand(1,heads,97,128) for _ in range(2)],value_cache=[rand(1,heads,97,128) for _ in range(2)])
    count=round(97*options['prefill_keep_ratio'])
    keep=[torch.arange(count,device='cuda').view(1,1,-1).expand(1,heads,-1) for _ in range(2)]
    packing={k:options[k] for k in ('prompt_bits','cold_bits','group_size','recent_window','lazy_g','decode_keep_ratio','optimizations')}
    cache=PhaseKVCache.from_dense_prefill(dense,prompt_len=97,keep_indices=keep,**packing)
    steps=options['recent_window']+2*options['lazy_g']+3
    worst=0.
    for step in range(steps+1):
        if step in (0,steps):
            for layer in cache.layers:
                assert layer.prompt.key.bits==layer.prompt.value.bits==options['prompt_bits']
                if step:
                    assert len(layer.cold)==2 and all(s.key.bits==s.value.bits==options['cold_bits'] for s in layer.cold)
                    assert all(s.length==round(16*options['decode_keep_ratio']) for s in layer.cold)
                segments=[layer.prompt,*layer.cold]
                ks=[unpack_dequantize(s.key) for s in segments];vs=[unpack_dequantize(s.value) for s in segments]
                if layer.hot_len:
                    hk,hv=layer.hot_views();ks.append(hk);vs.append(hv)
                k=torch.cat(ks,dim=2).repeat_interleave(4,dim=1)
                v=torch.cat(vs,dim=2).repeat_interleave(4,dim=1)
                q=rand(32*batch,128)
                expected=(torch.softmax((q[None,:,None,:].float()@k.float().transpose(-1,-2))/(128**.5),dim=-1)@v.float())[0,:,0,:]
                actual=segmented_phasekv_decode(q,layer,32*batch)
                torch.testing.assert_close(actual.float(),expected,atol=.003,rtol=.01)
                worst=max(worst,(actual.float()-expected).abs().max().item())
        if step<steps:
            for i in range(2):cache.append_hot(i,rand(1,heads,1,128),rand(1,heads,1,128))
            cache.finish_decode_step()
    assert cache.true_seq_length==97+steps
    return dict(max_absolute_attention_error=worst,batch=batch,append_steps=steps,cold_segments_per_layer=2,
        retained_tokens_per_cold_block=round(16*options['decode_keep_ratio']),tested_bits=[options['prompt_bits'],options['cold_bits']],
        reference='PyTorch attention over dequantized retained KV, before and after cooling/ring wrap')

def gate(engine,row,p,native_run):
    import torch
    import PhaseKV.kernels.segmented_attention as kernels
    class Count:
        def __init__(self,fn):self.fn=fn;self.count=0
        def __getitem__(self,grid):self.count+=1;return self.fn[grid]
    # The MMA JIT body calls _segmented_attention directly. Do not replace that
    # JIT dependency with a Python counter; instrument only the host launch.
    originals={name:getattr(kernels,name) for name in ('_a800_segmented_attention',)}
    counters={k:Count(v) for k,v in originals.items()}
    for name,value in counters.items():setattr(kernels,name,value)
    try:
        ids=row['input_ids'][:2112]
        measured=native_run(engine,[ids],3,[],True,{})
        check_measurement(measured,p,len(ids));measured.pop('output_ids');engine.release()
        details=synthetic_gate(c.options(p))
        launches=counters['_a800_segmented_attention'].count
        if (p['prompt_bits']==4 and launches==0) or (p['prompt_bits']==8 and launches!=0):
            raise ValueError('Unexpected A800 MMA dispatch for prompt bit width')
        return dict(status='ok',profile=p,model_probe=measured,synthetic=details,
            kernel_launches={k:v.count for k,v in counters.items()},
            generic_dispatch='8-bit packed descriptors use the existing generic segmented kernel; no claim of 4-bit MMA acceleration',
            scope='Dedicated gate inputs only, excluded from quality scores and speed measurements')
    finally:
        for name,value in originals.items():setattr(kernels,name,value)
        engine.release();gc.collect();torch.cuda.empty_cache()


def batch_gate(engine,bank,p,native_run):
    import torch
    tests=[r[:2112] for r in bank[:8]]
    if len(tests)!=8 or len({tuple(r) for r in tests})!=8:raise ValueError('Eight distinct probe requests required')
    def probe(rows):
        ids=torch.tensor(rows,device='cuda');first=engine.start(ids,2048,False,{})
        post=engine.step(ids[:,-8:-7].contiguous())
        evidence=dict(engine.evidence(),effective_options=engine.effective,decode_steps=1,batch=len(rows))
        check_measurement(evidence,p,len(rows[0]))
        result=torch.cat([first,post],1).float().cpu();engine.release();return result
    single=torch.cat([probe([r]) for r in tests]);errors={}
    for b in (1,2,4,8):
        current=probe(tests[:b]);den=single[:b].norm(dim=-1).clamp_min(1e-6)
        error=((current-single[:b]).norm(dim=-1)/den).max().item()
        if not torch.isfinite(current).all() or error>.04:raise ValueError(f'Batch independence failed B{b}: {error}')
        errors[str(b)]=error
    smoke=native_run(engine,tests[:8],164,[],False,{})
    check_measurement(smoke,p,2112);smoke.pop('output_ids');engine.release()
    return dict(status='ok',batch_logit_relative_l2=errors,b8_cooling_smoke=smoke,
        synthetic_b8=synthetic_gate(c.options(p),batch=8),scope='Excluded from timed runs and quality scores')

