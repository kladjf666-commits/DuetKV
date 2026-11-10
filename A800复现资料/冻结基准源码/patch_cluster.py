"""Apply only to the pinned ClusterKV checkout. Idempotent, checked against Git."""
from pathlib import Path
import subprocess,sys

PIN='c7380819588771e8c5dadacdbd33942d7c53adea'

def patch(root):
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args])
    if git('rev-parse','HEAD').decode().strip()!=PIN:raise ValueError('Unexpected ClusterKV revision')
    def update(name,transform):
        original=git('show',f'HEAD:{name}').decode().replace('\r\n','\n')
        expected=transform(original);p=root/name
        if expected==original or p.read_text(encoding='utf8') not in (original,expected):
            raise ValueError('Unexpected source in '+name)
        p.write_text(expected,encoding='utf8',newline='\n')
    def attn(s):
        start=s.index('    def _init_rope(');end=s.index('    def _shape',start)
        s=s[:start]+'''    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def a800_rope(self, query, key, position_ids):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        cos, sin = self.rotary_emb(key, position_ids)
        q, k = apply_rotary_pos_emb(query.unsqueeze(0), key.unsqueeze(0), cos, sin, unsqueeze_dim=2)
        return q.squeeze(0).contiguous(), k.squeeze(0).contiguous()

'''+s[end:]
        start=s.index('        clusterkv.utils.apply_rope_in_place(')
        end=s.index('        torch.cuda.nvtx.range_pop()',start)
        s=s[:start]+'''        if position_ids is None:
            position_ids = torch.arange(controller.kv_seqlen-q_len, controller.kv_seqlen, device=key_states.device)[None]
        query_states, key_states = self.a800_rope(query_states, key_states, position_ids)
'''+s[end:]
        s=s.replace('                    with torch.cuda.stream(controller.build_cluster_stream):','                    controller.build_cluster_stream.wait_stream(torch.cuda.current_stream())\n                    key_states.record_stream(controller.build_cluster_stream)\n                    with torch.cuda.stream(controller.build_cluster_stream):')
        return s.replace('wait(controller.build_cluster_stream)','wait(torch.cuda.current_stream())')
    def mask(s):
        a=s.index('        if attention_mask is None:',s.index('        # embed positions'));b=s.index('        hidden_states = inputs_embeds',a)
        return s[:a]+'''        # The unpadded B1 prefill kernel already applies its own causal mask.
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

'''+s[b:]
    update('kernel/CMakeLists.txt',lambda s:s.replace('CMAKE_CUDA_ARCHITECTURES 89','CMAKE_CUDA_ARCHITECTURES 80'))
    update('clusterkv/clusterkv_models/ClusterKVAttention.py',attn)
    update('clusterkv/clusterkv_models/llama.py',mask)
    from cluster_paper import extra
    extra(update)

def verify_rope(model):
    import torch,math
    a=model.model.layers[0].self_attn;cfg=model.config;dev=a.q_proj.weight.device
    n=a.head_dim;pos=torch.tensor([[0,1,8192,32767]],device=dev)
    q=torch.linspace(-1,1,4*3*n,device=dev).reshape(4,3,n).half();k=q[:,:1].contiguous()
    actual=a.a800_rope(q,k,pos)
    r=cfg.rope_scaling;orig=r['original_max_position_embeddings'];lo=r['low_freq_factor'];hi=r['high_freq_factor']
    freq=1/(cfg.rope_theta**(torch.arange(0,n,2,device=dev).float()/n))
    wave=2*math.pi/freq;smooth=(orig/wave-lo)/(hi-lo)
    inv=torch.where(wave>orig/lo,freq/r['factor'],torch.where(wave<orig/hi,freq,(1-smooth)*freq/r['factor']+smooth*freq))
    ang=(pos[0,:,None].float()*inv[None]).repeat(1,2)[:,None,:]
    errors=[]
    for inp,out in zip((q,k),actual):
        x,y=inp.chunk(2,-1);ref=inp*ang.cos().half()+torch.cat((-y,x),-1)*ang.sin().half()
        torch.testing.assert_close(out,ref,atol=.004,rtol=.004);errors.append((out-ref).abs().max().item())
    return max(errors)
if __name__=='__main__':patch(Path(sys.argv[1]))
