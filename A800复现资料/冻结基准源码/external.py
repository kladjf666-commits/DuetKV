"""B1 adapters, using the pinned upstream implementations, one process per method."""
import contextlib,hashlib,json,os,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
def load(name,path):
    from transformers import AutoConfig,AutoModelForCausalLM
    cfg=AutoConfig.from_pretrained(path,local_files_only=True)
    assert (cfg.num_attention_heads,cfg.num_key_value_heads,cfg.rope_scaling.get('rope_type',cfg.rope_scaling.get('type')))==(32,8,'llama3')
    if name=='million':
        sys.path.insert(0,str(HERE/'upstream/MILLION'))
        from scripts.utils.Namespace import UniConfig
        conf=UniConfig();conf.nbits=8;conf.breakdown=False;conf.distort_recent=False
        from scripts.modeldb.models.ModelContext import get_context
        from scripts.utils.pq_utils import DynamicPQCache
        ctx=get_context('llama');bookdir=Path(os.environ['A800_OUT'])/'codebook'
        meta=json.loads((bookdir/'meta.json').read_text())
        if meta['model_config']!=hashlib.sha256((path/'config.json').read_bytes()).hexdigest() or meta['weights_index']!=hashlib.sha256((path/'model.safetensors.index.json').read_bytes()).hexdigest():raise ValueError('Codebook belongs to different weights')
        books=[]
        for side in ('key','value'):
            f=bookdir/(side+'.pt');assert hashlib.sha256(f.read_bytes()).hexdigest()==meta[side]
            arr=torch.load(f,weights_only=True,map_location='cpu')
            assert arr.shape==(64,256,2) and torch.isfinite(arr).all()
            books.append(arr.cuda().half())
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
        cache=DynamicPQCache(bs=1,nh=32,num_key_value_heads=8,M=64,layer_num=32,dtype=torch.uint8,nbits=8,d=128,scalar_t=torch.float16)
        cache.set_cent(*books)
        ctx.evaluation_context.__enter__() # lifetime is this single-method process
        assert model.model.layers[0].self_attn.forward.__name__=='attn_forward_custom_kernel'
        def forward(ids,start):
            pos=torch.arange(start,start+ids.shape[1],device=ids.device)
            out=model.model(input_ids=ids,position_ids=pos[None],cache_position=pos,use_cache=False)
            return model.lm_head(out.last_hidden_state[:,-1:]).float()
        def evidence():
            assert all(t.size(2)>0 for t in cache.key_cache)
            return dict(trace=dict(pq_tokens=cache.key_cache[0].size(2),code_dtype=str(cache.key_cache[0].dtype)),final_kv_bytes=sum(t.numel()*t.element_size() for arrays in (cache.key_cache,cache.value_cache,cache.key_residual_cache,cache.value_residual_cache) for t in arrays))
        ordinary=cache.decoding
        def reset(l,n,quality):
            cache.init_cache();cache.decoding=ordinary
            if quality:
                from million_quality import set_zero_residual
                set_zero_residual(cache)
        return model,dict(reset=reset,release=cache.init_cache,forward=forward,evidence=evidence,settings=dict(M=64,nbits=8,bits_per_scalar=4,quality_residual=0,speed_residual=128,codebooks=meta))
    sys.path.insert(0,str(HERE/'upstream/ClusterKV'))
    from clusterkv.clusterkv_models.llama import LlamaForCausalLM
    model=LlamaForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,attn_implementation='eager').cuda().eval()
    def release():model.model.controller=None
    def reset(l,n,quality):
        torch.manual_seed(20261029)
        initial=max(1,l//80);budget=256 if quality else 1024
        if initial+((n-1)//320)*4>512 or l<16+budget:raise ValueError('Input exceeds validated paper profile')
        release();model.clusterkv_init(nlist=initial,niter=20,max_seq_len=l+n+4,token_budget=budget,
            dtype=torch.float16,device=torch.device('cuda:0'),full=False,sink=16,window=320,window_nlist=4,offload=True)
    def forward(ids,start):
        pos=torch.arange(start,start+ids.shape[1],device=ids.device)[None]
        out=model.model(input_ids=ids,position_ids=pos,use_cache=False,return_dict=True)
        return model.lm_head(out.last_hidden_state[:,-1:]).float()
    def evidence():
        c=model.model.controller
        assert not c.full and c.offload and c.centroids[2] is not None
        return dict(trace=dict(physical_history=c.kv_seqlen,retrieval_budget=c._token_budget,initial_clusters=c.nlist,clusters=c.centroids[2].shape[1],cpu_pinned_kv_bytes=sum(t.numel()*t.element_size() for t in c.kv_cache_cpu if t is not None)),final_kv_bytes=sum(t.numel()*t.element_size() for t in c.kv_cache))
    return model,dict(reset=reset,release=release,forward=forward,evidence=evidence,settings=dict(quality_budget=256,speed_budget=1024,nlist='floor(L/80)',niter=20,sink=16,window=320,window_nlist=4,offload=True))
