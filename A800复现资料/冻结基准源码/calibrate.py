"""Fit separate key/value PQ dictionaries from WikiText-2 raw training inputs."""
import gzip,hashlib,json,os,sys
from pathlib import Path
import torch,faiss
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
here=Path(__file__).resolve().parent;sys.path.insert(0,str(here/'upstream/MILLION'))
from scripts.utils.pq_utils import train_pq
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
@torch.inference_mode()
def main():
    from worker import hw,write
    hw()
    path=Path(os.environ['PHASEKV_MODEL']);dest=Path(os.environ['A800_OUT'])/'codebook';dest.mkdir(parents=True,exist_ok=True)
    source=here/'data/wikitext2_train.parquet';meta=dest/'meta.json'
    info=dict(model_config=sha(path/'config.json'),weights_index=sha(path/'model.safetensors.index.json'),calibration_sha256=sha(source),seed=42,M=64,nbits=8,recipe='WikiText-2-raw-v1 train only; 16 seeded 4096-token windows; 128 vectors per layer per window; 65536 vectors per side; separate K/V dictionaries shared across layers; FAISS IndexPQ 25 iterations')
    if meta.exists():
        prev=json.loads(meta.read_text())
        if all(prev.get(k)==v for k,v in info.items()) and all(sha(dest/(k+'.pt'))==prev[k] for k in ('key','value')):
            print('Verified existing codebooks');return
        raise ValueError('Existing codebooks differ. Use a new A800_OUT directory.')
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    rng=torch.Generator().manual_seed(info['seed']);keys=[];values=[]
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    texts=pq.read_table(source,columns=['text']).column('text').to_pylist()
    bank=tokenizer.encode('\n\n'.join(texts),add_special_tokens=False)
    blocks=torch.randperm(len(bank)//4096,generator=rng)[:16].tolist()
    assert len(blocks)==16
    info['blocks']=blocks
    for i in blocks:
        ids=torch.tensor([bank[i*4096:(i+1)*4096]],device='cuda')
        cache=DynamicCache();model.model(input_ids=ids,past_key_values=cache,use_cache=True)
        for k,v in zip(cache.key_cache,cache.value_cache):
            k=k.reshape(-1,128);v=v.reshape(-1,128);sel=torch.randint(len(k),(128,),generator=rng).to(k.device)
            keys.append(k[sel].cpu());values.append(v[sel].cpu())
        del cache
    del model;torch.cuda.empty_cache();faiss.omp_set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    for name,rows in [('key',keys),('value',values)]:
        vectors=torch.cat(rows).float().numpy();assert vectors.shape[0]>=60000
        book=train_pq(vectors,64,8);assert book.shape==(64,256,2) and torch.isfinite(book).all()
        p=dest/(name+'.pt');tmp=dest/(name+'.tmp');torch.save(book.contiguous(),tmp);tmp.replace(p);info[name]=sha(p)
        info['vectors_per_codebook']=vectors.shape[0]
    write(meta,info);print('Calibrated codebooks',info)
if __name__=='__main__':main()
