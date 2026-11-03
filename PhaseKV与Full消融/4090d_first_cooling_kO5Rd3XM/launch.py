import json, sys, time, os
from pathlib import Path
out=Path(sys.argv[1])
sys.path.insert(0,'/data/phasekv_4090d_speed')
import quick
import torch
import native_runtime, ablation_runtime as rt, ablation_kernels
from PhaseKV.kernels.segmented_attention import segmented_phasekv_decode
from unittest.mock import patch
sample,checksum=quick.verify_bundle()
historical=json.loads(Path('/data/phasekv-results/4090d_speed_20261101_221816_361/summary.json').read_text())
env,device=quick.gpu_environment(torch)
assert checksum==historical['protocol']['code_sha256'] and device.split(',')[0]==historical['protocol']['gpu'].split(',')[0] and env==historical['protocol']['environment']
opts=dict(historical['protocol']['options_gqa_off'])
opts['optimizations']=dict(opts['optimizations'],gqa_reuse=True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
print('Loading same 32-layer Llama checkpoint on',device,flush=True)
with torch.inference_mode(),patch.object(ablation_kernels,'segmented_phasekv_decode',segmented_phasekv_decode):
    model,_,_=rt.load_model(Path(historical['protocol']['model']),'full_fa2')
    assert len(model.model.layers)==32
    print('MODEL_READY',flush=True)
    end=time.monotonic()+900
    while not (out/'measure.py').is_file():
        if time.monotonic()>end: raise RuntimeError('Timed out waiting for measurement script')
        time.sleep(1)
    ns={'__name__':'cooling_probe','__file__':str(out/'measure.py')}
    exec(compile((out/'measure.py').read_text(),str(out/'measure.py'),'exec'),ns)
    ns['run'](model,sample,opts,historical,out)
