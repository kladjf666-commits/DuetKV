"""Build the independent host-timing overlay; leave earlier packages intact."""
import hashlib
import zipfile
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    files = {name: (here / source).read_bytes().replace(b'\r\n', b'\n') for name, source in (
        ('cooling_cpu.py', 'cooling_cpu.py'), ('start.sh', 'start_cpu.sh'), ('README.txt', 'README_cpu.txt'))}
    files['SHA256SUMS'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
                                 for name, data in files.items()).encode()
    destination = root / 'PhaseKV与Full消融/PhaseKV_4090D_cooling_cpu.zip'
    temporary = destination.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr('phasekv_4090d_cooling_cpu/' + name, data)
    temporary.replace(destination)
    print(destination, destination.stat().st_size)


if __name__ == '__main__':
    build()
