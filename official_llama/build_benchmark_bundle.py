"""Package only additive formal-test files; do not overwrite validated runtimes."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent
NAMES=['README.md','benchmark.py','benchmark_generate.py','benchmark_metrics.py',
       'prepare_benchmark.py','fetch_ruler_essays.py','start_benchmark.sh',
       'verify_benchmark_bundle.py','build_benchmark_bundle.py','tests/test_benchmark.py',
       'tests/check_frozen_benchmark.py','validation/benchmark_delivery_checks.json',
       'validation/benchmark_tests.txt','validation/benchmark_frozen_checks.json',
       'bench_assets/essay_sources.json']


def main():
    files={ROOT/name for name in NAMES}
    for folder in ('bench_assets/frozen','bench_assets/wheels','bench_sources/RULER'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and '.git' not in p.parts and '__pycache__' not in p.parts)
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    manifest=ROOT/'benchmark_bundle.json'
    manifest.write_text(json.dumps({'date':'2026-10-29','files':hashes},indent=2)+'\n',encoding='utf-8')
    files.add(manifest)
    target=ROOT/'delivery/PhaseKV_formal_benchmark_20261029.zip'
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files):
            z.write(path,'official_llama/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,expected in hashes.items():
            assert hashlib.sha256(z.read('official_llama/'+name)).hexdigest()==expected
    sha=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(sha+'  '+target.name+'\n',encoding='ascii')
    print(json.dumps({'path':str(target),'bytes':target.stat().st_size,'sha256':sha,'files':len(files)},indent=2))


if __name__=='__main__':
    main()
