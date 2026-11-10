"""Use the unchanged frozen speed implementation with the new four-shape matrix."""
from pathlib import Path
import argparse
import sys
import traceback
from common import HERE,SHAPES,read,verify,weight_identity,speed_cases,speed_result

def execute(config_path,method):
    config=read(config_path)
    if verify(HERE)!=config['manifest']:raise ValueError('Recovery package changed')
    if weight_identity(config['model'])!=config['weights']:raise ValueError('Model files changed')
    for case,b in speed_cases(config['bench'],method):speed_result(config['old_out'],method,case,b)
    sys.path.insert(0,config['bench'])
    import torch
    import worker
    args=argparse.Namespace(action='speed',method=method,rounds=3,shapes=list(SHAPES),
        batches=[1] if method in ('million','clusterkv') else [1,2,4,8],data_groups=None,suites=None,limit=None)
    try:
        with torch.inference_mode():worker.work(args)
        (worker.OUT/method/'speed_last_error.json').unlink(missing_ok=True)
    except Exception as exc:
        worker.write(worker.OUT/method/'speed_last_error.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--method',required=True)
    a=p.parse_args();execute(a.config,a.method)
