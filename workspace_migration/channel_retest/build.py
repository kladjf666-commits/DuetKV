"""Package the tested source files without replacing the frozen server installation."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build(target):
    files = {n: (HERE/n).read_bytes() for n in ('retest.py','run.sh')}
    files['historical_summary.json'] = (ROOT/'PhaseKV与Full消融/summary.json').read_bytes()
    for n in ('probe.py','batch_attention.py','trace_budget.py'):
        files[n] = (HERE.parent/'decode_probe'/n).read_bytes()
    files['channel_impl/__init__.py'] = b'"""Isolated channel-contiguous K runtime."""\n'
    base = ROOT/'PhaseKV'
    for path in [base/n for n in ('optimizations.py','phasekv_cache.py','packed_storage.py')]+sorted((base/'kernels').glob('*.py')):
        files['channel_impl/'+path.relative_to(base).as_posix()] = path.read_bytes()
    files = {k:v.replace(b'\r\n',b'\n') for k,v in files.items()}
    hashes = {k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    files['files.json'] = json.dumps(hashes,indent=2).encode()
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for k,v in files.items():
            z.writestr('migration/channel_only/'+k,v)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for k,v in files.items():
            assert z.read('migration/channel_only/'+k)==v


if __name__=='__main__':
    import sys
    build(Path(sys.argv[1]))
