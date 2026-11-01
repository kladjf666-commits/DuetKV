"""Offline Triton 3.1 compilation for common FP16 Llama 8192+256 variants.

Run on the CPU instance *after* staging the Linux wheels. Produces native cache
entries on the persistent volume using the exact installed compiler and paths.
Compilation is not execution, numerical validation, or a guarantee of coverage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path('/root/data/phasekv-ws/official_llama'))
    parser.add_argument('--arch',type=int,choices=(80,89),default=80,
                        help='80: A800, 89: RTX 4090. Use separate caches on the volume.')
    parser.add_argument('--cache',type=Path,help='Defaults to /root/data/phasekv-triton-sm<arch>')
    parser.add_argument('--sms',type=int,help='SM count for launch planning; defaults to 108 (A800) or 128 (RTX 4090)')
    args=parser.parse_args()
    sms=args.sms if args.sms is not None else {80:108,89:128}[args.arch]
    if sms<1: parser.error('--sms must be positive')
    project=args.project.resolve()
    cache=(args.cache or Path(f'/root/data/phasekv-triton-sm{args.arch}')).resolve()
    cache.mkdir(parents=True,exist_ok=True)
    os.environ['TRITON_CACHE_DIR']=str(cache)
    sys.path[:0]=[str(project/'.cp310-site'),str(project/'.torch-cu121'),str(project),str(project/'runtime')]
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.runtime.jit import MockTensor
    from PhaseKV.kernels.launch_config import plan_kivi_decode, plan_phasekv_cold_group
    from PhaseKV.kernels.segmented_plan import DESCRIPTOR_WIDTH, _capacity
    from PhaseKV.kernels.packing import allocate_packed
    from PhaseKV.kernels.quantize_triton import _quantize_4bit
    from ablation_kernels import _segmented_attention, _merge_segments, _unpack_segment
    if triton.__version__!='3.1.0': raise RuntimeError('Requires the pinned Triton 3.1.0')
    target=GPUTarget('cuda',args.arch,32)
    seen=set()
    binaries={}

    def emit(fn, *values, **kwargs):
        # Use the same 3.1 argument binder and specialization attributes as
        # JITFunction.run; compile for an explicit target, without a driver.
        if fn.binder is None: fn.create_binder()
        values=[MockTensor(v.dtype) if hasattr(v,'dtype') else v for v in values]
        bound,sigs,_,_,extra=fn.binder(*values,**kwargs,debug=fn.debug)
        vals=tuple(bound.values())
        config=fn._get_config(*vals)
        names=[fn.params[i].name for i in fn.non_constexpr_indices]
        signature=dict(zip(names,sigs[:len(names)]))
        constants={p.name:v for p,v in zip(fn.params,vals) if p.is_constexpr or p.num in config.equal_to_1 or v is None}
        key=(fn.__name__,tuple(signature.items()),tuple(constants.items()),tuple(sorted(config.divisible_by_16)),
             tuple(sorted(config.equal_to_1)),tuple(sorted(extra.items())))
        if key in seen: return
        options=fn.make_backend(target).parse_options(extra)
        source=fn.ASTSource(fn,signature,constants,config)
        binary=fn.compile(source,target=target,options=options.__dict__)
        if binary.metadata.target.arch!=args.arch or not binary.kernel.startswith(b'\x7fELF'):
            raise RuntimeError(f'Compiler did not produce an SM{args.arch} ELF cubin')
        seen.add(key)
        binaries[binary.hash]={'kernel':fn.__name__,'sha256':hashlib.sha256(binary.kernel).hexdigest()}

    fp16=torch.empty(0,dtype=torch.float16)
    i64=torch.empty(0,dtype=torch.int64)
    fp32=torch.empty(0,dtype=torch.float32)
    h,hd,group=8,128,4
    prompt=8192
    retained=round(prompt*.7)
    cap=144
    def launch_parts(length,count,capacity,packed,hot_len,**schedule):
        emit(_segmented_attention,fp16,i64,fp16,fp16,fp32,fp32,fp32,
             packed,count,hot_len,hd,1,length*hd,hd,1,length*hd,hd,1,
             HD=hd,GROUP=group,BN=8,DW=DESCRIPTOR_WIDTH,CAPACITY=capacity,
             num_warps=4,enable_fp_fusion=False,**schedule)
        emit(_merge_segments,fp32,fp32,fp32,fp16,count,HD=hd,BP=16,CAPACITY=capacity,
             num_warps=4,enable_fp_fusion=False)
    pieces={}
    for length in (retained,16):
        pk,pv=[allocate_packed((1,h,length,hd),4,axis,32,'meta') for axis in (2,3)]
        pieces[length]=pk,pv
        source_len=cap if length==16 else length
        x=torch.empty((1,h,source_len,hd),device='meta',dtype=torch.float16)
        for pt in (pk,pv):
            emit(_quantize_4bit,x,pt.packed,pt.scale,pt.mins,*x.stride(),h,length,hd,source_len,0,
                 pt.axis,pt.axis_len,pt.group_size,pt.scale.shape[-1],*pt.packed.stride(),
                 *pt.scale.stride(),*pt.mins.stride(),BLOCK=max(2,triton.next_power_of_2(pt.group_size)),
                 num_warps=4,enable_fp_fusion=False)
    prefix=plan_kivi_decode(retained,32,hd,sms).num_splits
    cold_splits=plan_kivi_decode(16,32,hd,sms).num_splits
    for step in range(1,256):
        length=prompt+step
        dense=plan_kivi_decode(length,32,hd,sms)
        launch_parts(length,dense.num_splits,_capacity(dense.num_splits,True),0,length,
                     DENSE_MODE=True,DENSE_SPLIT=dense.split_len,QPER=1)
        cooled=max(0,(step-1-128)//16)
        cg=plan_phasekv_cold_group(cooled,8)
        rows=cooled*(cold_splits if cg==1 else 1)
        count=prefix+(rows+cg-1)//cg+1
        launch_parts(cap,count,_capacity(count,True),count-1,step-16*cooled,
                     PREFIX=prefix,COLD_GROUP=cg,NROWS=prefix+rows,FOUR_BIT=True,
                     QPER=1,HOT_CAP=cap,HOT_START=(16*cooled)%cap)
        for n,offset in [(retained,0)]+([(16,retained)] if cooled else []):
            k,v=pieces[n]
            emit(_unpack_segment,k.packed,k.scale,k.mins,v.packed,v.scale,v.mins,fp16,fp16,
                 n,hd,h,retained+step,offset,4,4,k.group_size,v.group_size,
                 *k.packed.stride()[1:],*k.scale.stride()[1:],*v.packed.stride()[1:],*v.scale.stride()[1:],
                 BLOCK=256,enable_fp_fusion=False)
    report={'target':f'sm{args.arch}','sm_count_hint':sms,'triton':triton.__version__,
            'input':prompt,'output':256,'batch':1,'gpu_execution':False,'binaries':binaries}
    (cache/'prewarm_8192.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Compiled/verified {len(binaries)} SM{args.arch} variants. No GPU execution. Cache: {cache}')


if __name__=='__main__':
    main()
