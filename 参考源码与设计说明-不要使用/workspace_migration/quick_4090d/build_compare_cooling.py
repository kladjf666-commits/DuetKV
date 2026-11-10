"""Build the two-file PhaseKV overlay and its same-process 4090D comparison."""
import hashlib
import zipfile
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[1]
files = {name: path.read_bytes().replace(b'\r\n', b'\n') for name, path in (
    ('phasekv_cache.py', root / 'PhaseKV/phasekv_cache.py'),
    ('segmented_plan.py', root / 'PhaseKV/kernels/segmented_plan.py'),
    ('cooling_cpu.py', here / 'cooling_cpu.py'), ('compare_cooling.py', here / 'compare_cooling.py'),
    ('start.sh', here / 'start_compare_cooling.sh'))}
files['SHA256SUMS'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
                             for name, data in files.items()).encode()
target = root / 'PhaseKV与Full消融/PhaseKV_4090D_cooling_ab.zip'
temp = target.with_suffix('.tmp')
with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as dest:
    for name, data in files.items():
        dest.writestr('phasekv_4090d_cooling_ab/' + name, data)
temp.replace(target)
print(target, target.stat().st_size)
