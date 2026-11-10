"""Package only the authoritative core and quick test; no historical duplicate."""
from pathlib import Path
import argparse, ast, hashlib, json, tarfile

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'deliverables')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / name for name in ('README.md', 'BETTER_MEMORY_QUICK.txt', 'run.sh', 'validation.json')]
    for directory in ('PhaseKV', 'benchmark', 'data', 'tests'):
        paths.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    paths = sorted(paths)
    for path in paths:
        if path.suffix == '.sh':
            path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
        if path.suffix == '.py': ast.parse(path.read_text('utf-8'), feature_version=(3, 10))
    provenance = json.loads((ROOT / 'benchmark/provenance.json').read_text('utf-8'))
    assert sha(ROOT / 'benchmark/baseline_runtime.py') == provenance['baseline_runtime_sha256']
    for name, checksum in provenance['core_source_manifest'].items():
        if name != 'phasekv_cache.py': assert sha(ROOT / 'PhaseKV' / name) == checksum, name
    validation = json.loads((ROOT / 'validation.json').read_text('utf-8'))
    for name, checksum in validation['tested_source_sha256'].items():
        assert sha(ROOT / name) == checksum, 'Source changed after CPU validation: ' + name
    manifest = ROOT / 'SHA256SUMS'
    manifest.write_bytes(''.join(sha(p) + '  ' + p.relative_to(ROOT).as_posix() + '\n' for p in paths).encode('utf-8'))
    paths.append(manifest)
    archive = args.output / 'a800_bettermemory_quick_v1.tar.gz'
    if archive.exists(): raise FileExistsError('Do not overwrite a delivered package: ' + str(archive))
    with tarfile.open(archive, 'w:gz') as tar:
        for path in paths: tar.add(path, arcname='a800_bettermemory_quick_v1/' + path.relative_to(ROOT).as_posix())
    archive.with_name(archive.name + '.sha256').write_bytes((sha(archive) + '  ' + archive.name + '\n').encode('ascii'))
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers()) == len(paths)
        for path in paths:
            assert tar.extractfile('a800_bettermemory_quick_v1/' + path.relative_to(ROOT).as_posix()).read() == path.read_bytes()
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                         sha256=sha(archive), files=len(paths)), ensure_ascii=True, indent=2))

if __name__ == '__main__': main()
