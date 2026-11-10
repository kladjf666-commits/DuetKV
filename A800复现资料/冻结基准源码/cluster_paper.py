"""Source adapters needed for the paper's L/80 clusters and host recall on SM80."""
def extra(update):
    def rank(s):
        s=s.replace('shared_dist[256]','shared_dist[512]').replace('shared_indices[256]','shared_indices[512]')
        s=s.replace('std::numeric_limits<DistT>::min()','static_cast<DistT>(-INFINITY)')
        s=s.replace('size <= nlist; size *= 2','size <= blockDim.x; size *= 2').replace('j > idx && j < nlist','j > idx && j < blockDim.x')
        s=s.replace('if (shared_dist[idx] < shared_dist[j])','if (((idx & size) == 0 && shared_dist[idx] < shared_dist[j]) || ((idx & size) != 0 && shared_dist[idx] > shared_dist[j]))')
        old='int tile_sum = __shfl_sync(0xFFFFFFFF, scan_output, block_size - 1);\n            running_prefix += tile_sum;'
        assert old in s
        return s.replace(old,'if (tile_offset + block_size < nlist) running_prefix = sh_neigh_size_ps[tile_offset + block_size - 1];\n            __syncthreads();')
    update('kernel/cluster/include/km_include/cluster.cuh',rank)
    def launch(s):
        a=s.index('const int block_size = 256;',s.index('void get_neigh_c'))
        return s[:a]+s[a:].replace('const int block_size = 256;', 'TORCH_CHECK(nlist > 0 && nlist <= 512, "A800 cluster count outside validated range");\n    const int block_size = nlist <= 256 ? 256 : 512;',1)
    update('kernel/cluster/src/cluster.cu',launch)
    def norm(s):
        s=s.replace('controller.update_metadata(layer_idx, centroids,','controller.update_metadata(layer_idx, centroids_normalized.contiguous(),')
        return s.replace('            _kernels.recall(', '            controller.default_stream.wait_event(controller.offload_events[layer_idx])\n            _kernels.recall(')
    update('clusterkv/clusterkv_utils/__init__.py',norm)
    def streams(s):
        s=s.replace('with torch.cuda.stream(self.offload_stream):\n\t\t\tself.kv_cache_cpu[layer_idx][:seq_len', 'self.offload_stream.wait_stream(torch.cuda.current_stream())\n\t\tk.record_stream(self.offload_stream)\n\t\tv.record_stream(self.offload_stream)\n\t\twith torch.cuda.stream(self.offload_stream):\n\t\t\tself.kv_cache_cpu[layer_idx][:seq_len')
        return s.replace('with torch.cuda.stream(self.offload_stream):\n\t\t\tself.kv_cache_cpu[layer_idx][self.kv_seqlen-window:', 'self.offload_stream.wait_stream(torch.cuda.current_stream())\n\t\tkv.record_stream(self.offload_stream)\n\t\twith torch.cuda.stream(self.offload_stream):\n\t\t\tself.kv_cache_cpu[layer_idx][self.kv_seqlen-window:')
    update('clusterkv/clusterkv_utils/clusterkv_controller.py',streams)

def verify_rank():
    import torch
    from clusterkv import _clusterkv_knl as kn
    # Exact FP16 scores, arbitrary ordering, non-power-of-two count, and a budget in the second scan tile.
    for n in (26,409,457):
        q=torch.zeros(1,32,128,device='cuda',dtype=torch.float16);q[:,:,0]=1
        cent=torch.zeros(8,n,128,device='cuda',dtype=q.dtype)
        perm=torch.randperm(n,device='cuda');cent[:,:,0]=perm.half()
        counts=torch.full((8,n),2,device='cuda',dtype=torch.int32)
        neigh=torch.empty(32,n,device='cuda',dtype=torch.int32);sizes=torch.empty_like(neigh)
        kn.get_neigh_c(q,cent,counts,neigh,sizes)
        order=perm.argsort(descending=True);assert torch.equal(neigh.long(),order.expand(32,-1))
        ps=counts.cumsum(1,dtype=torch.int32)
        inds=torch.arange(n*2,device='cuda',dtype=torch.int32).repeat(8,1)
        pick=torch.empty(32,min(631,n*2-1),device='cuda',dtype=torch.int32)
        kn.get_sel_indices(neigh,sizes,ps.contiguous(),inds,pick)
        ref=torch.stack([2*order,2*order+1],-1).flatten()[:pick.shape[-1]]
        assert torch.equal(pick.long(),ref.expand(32,-1))
    return True
