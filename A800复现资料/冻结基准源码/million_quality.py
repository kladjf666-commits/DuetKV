"""MILLION paper §IV-B stress setting: encode every decode KV immediately."""
def set_zero_residual(cache):
    from scripts.utils.pq_utils import sa_encode_4d_keops
    def step(q,k,v,i):
        ck=sa_encode_4d_keops(k,cache.key_cent,target_dtype=cache.dtype)
        cv=sa_encode_4d_keops(v,cache.value_cent,target_dtype=cache.dtype)
        cache.cat_codes(ck,cv,i)
        op=cache.registery.get_kernel(l=cache.seen_tokens[i])
        return op(q,cache.key_cache[i],cache.value_cache[i],cache.key_cent,cache.value_cent,cache.key_residual_cache[i],cache.value_residual_cache[i],0)
    cache.decoding=step

def patch_build(path):
    p=path/'Kernel.cuh';s=p.read_text()
    start=s.index('__global__ void flash_decoding_residual_kernel');pos=s.index('const cuscalar_t scale',start)
    block='''if (r == 0) {
        out[((b * nh + h) * (Ns + 1) + Ns) * d + tid] = cs::to_cuscalar<cuscalar_t>(0.0f);
        if (tid == 0) lse[(b * nh + h) * (Ns + 1) + Ns] = cs::to_cuscalar<cuscalar_t>(-CUDART_INF_F);
        return;
    }
    '''
    if block not in s:p.write_text(s[:pos]+block+s[pos:])
if __name__=='__main__':
    import sys
    from pathlib import Path
    patch_build(Path(sys.argv[1]))
