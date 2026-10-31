"""Portable upload bundle with Git-pinned source blobs; no weights or caches."""
from pathlib import Path
import hashlib
import json
import zipfile
from prepare import ROOT, verify


def main():
    manifest = verify()
    output = ROOT / 'delivery/PhaseKV_native_llama_20261028.zip'
    output.parent.mkdir(exist_ok=True)
    files = {ROOT / name for name in manifest['files']}
    files.add(ROOT / 'sources.json')
    files.update(ROOT.glob('*.py'))
    files.update(ROOT.glob('*.sh'))
    files.update(ROOT.glob('*.txt'))
    files.update(ROOT.glob('*.json'))
    files.update((ROOT / 'tests').glob('*.py'))
    files.update((ROOT / 'validation').glob('*.txt'))
    files.update((ROOT / 'validation').glob('*.json'))
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT.parent).as_posix())
    with zipfile.ZipFile(output) as archive:
        for relative, expected in manifest['files'].items():
            data = archive.read('official_llama/' + relative)
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('Bundle fingerprint mismatch: ' + relative)
        if archive.testzip() is not None:
            raise RuntimeError('Bundle CRC error')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(digest + '  ' + output.name + '\n', encoding='ascii')
    print(json.dumps({'file': str(output), 'bytes': output.stat().st_size,
                      'sha256': digest, 'files': len(files)}, indent=2))


if __name__ == '__main__':
    main()
