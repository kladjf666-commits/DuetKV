"""One model/algorithm per worker. No hidden request-level loop in the timed forward."""
import os,sys,json,gc,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'core'),str(HERE/'native')]
import PhaseKV
assert Path(PhaseKV.__file__).resolve().parent==HERE/'core/PhaseKV'
from transformers.cache_utils import DynamicCache
import native_runtime as rt

class Engine:
    def __init__(self,name,path):
        self.name=name; self.path=path; self.extra={}
        if name in ('clusterkv','million'):
            from external import load
            self.model,self.extra=load(name,path)
            self.options=self.extra['settings']
        elif name in ('full','phasekv','pyramidkv'):
            import ablation_runtime as art, ablation_kernels
            from PhaseKV.kernels.segmented_attention import segmented_phasekv_decode
            ablation_kernels.segmented_phasekv_decode=segmented_phasekv_decode
            self.model,self.options,_=art.load_model(path,'phasekv_triton' if name=='phasekv' else 'full_fa2')
            if name=='phasekv':
                cfg=json.loads((HERE/'native/config.json').read_text())['methods']['phasekv']
                self.options=dict(cfg,decode_keep_ratio=.8,optimizations=dict(cfg['optimizations'],gqa_reuse=True,channel_contiguous_k=True))
            if name=='pyramidkv':self.options=dict(max_capacity_prompt=512,window_size=8,kernel_size=7,beta=20,pooling='maxpool')
        else:self.model,self.options,_=rt.load_model(path,name)
        self.model.eval()
    def start(self,ids,limit,quality,progress):
        name=self.name;length=ids.shape[1];self.offset=length;progress['stage']='prefill'
        if self.extra:
            self.extra['reset'](length,limit,quality)
            self.cache=None
            return self.extra['forward'](ids,0)
        options=dict(self.options)
        if name=='phasekv': options['prefill_keep_ratio']=.95 if quality else 1.
        self.effective,self.mode=rt.reset_sample(self.model,name,options,limit)
        self.cache=DynamicCache() if name!='kivi' else None
        if name=='phasekv' and quality:
            logits,self.cache=rt.phase_prefill(self.model,ids,options)
        elif name=='pyramidkv':
            from pyramid import prefill
            logits,self.cache=prefill(self.model,ids,options)
        elif name in ('phasekv','full'):
            from batch_backend import packed_prompt
            for i in range(0,length,512):
                logits,self.cache,_=rt.forward(self.model,ids[:,i:i+512],self.cache,i,name)
            if name=='phasekv':
                progress['stage']='pack_prompt';self.cache=packed_prompt(self.cache,length,options)
        else:
            logits,self.cache,_=rt.forward(self.model,ids,self.cache,0,name)
        return logits
    def step(self,token):
        if self.extra:
            logits=self.extra['forward'](token,self.offset)
        else:
            logits,self.cache,_=rt.forward(self.model,token,self.cache,self.offset,self.name)
            if self.name=='phasekv': self.cache.finish_decode_step()
        self.offset+=1
        return logits
    def evidence(self):
        if self.extra:return self.extra['evidence']()
        trace=rt.cache_trace(self.cache,self.name)
        if self.name=='pyramidkv':trace['layer_lengths']=[k.shape[-2] for k in self.cache.key_cache]
        return dict(trace=trace,final_kv_bytes=rt.cache_report(self.cache)['physical_bytes'])
    def release(self):
        self.cache=None
        if self.extra:self.extra['release']()
        gc.collect();torch.cuda.empty_cache()

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
        effective_options=getattr(e,'effective',e.options),mode=getattr(e,'mode','native'),**e.evidence())
