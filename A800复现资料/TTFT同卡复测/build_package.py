"""Validate and archive the isolated TTFT retest without editing frozen sources."""
from pathlib import Path
import ast, hashlib, importlib.metadata, json, os, subprocess, sys, tarfile

HERE=Path(__file__).resolve().parent
PACKAGE=HERE/'a800_ttft_ab_v1'
PROJECT=HERE.parents[1]
SOURCE=PROJECT/'Code-Original-NoChanging - A800 - BetterMemory'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    archive=HERE/'a800_ttft_ab_v1.tar.gz'
    if archive.exists():raise FileExistsError('Refuse to overwrite a delivered archive')
    provenance=json.loads((PACKAGE/'data/core_provenance.json').read_text('utf-8'))
    for name,digest in provenance.items():
        assert sha(PACKAGE/name)==digest==sha(SOURCE/name),name
    py_files=sorted(PACKAGE.rglob('*.py'))
    for p in py_files:ast.parse(p.read_text('utf-8'),filename=str(p),feature_version=(3,10))
    for p in PACKAGE.glob('*.sh'):assert b'\r' not in p.read_bytes()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    tested=subprocess.run([sys.executable,'-B',str(PACKAGE/'tests/test_ttft_retest.py')],
                          env=env,text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(tested.stdout)
    if tested.returncode:raise RuntimeError('CPU tests failed')
    validation=dict(status='passed',tests=6,python=sys.version,torch=importlib.metadata.version('torch'),
        transformers=importlib.metadata.version('transformers'),cuda_server_validation='pending',
        scope=['I/O guard and immutable checkpoints','fixed two-case inputs and balanced arm order',
               'idempotent prepare and changed-plan rejection','pair validation and separation of diagnostics from formal statistics',
               'small real Llama exact logits/packed KV before and after instrumentation for B1/B2/B8',
               'nested phase accounting and restoration after exceptions'],
        tested_source_sha256={p.relative_to(PACKAGE).as_posix():sha(p) for p in py_files})
    (PACKAGE/'validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    (PACKAGE/'validation.txt').write_text(tested.stdout,encoding='utf-8')
    paths=sorted(p for p in PACKAGE.rglob('*') if p.is_file() and p.name!='SHA256SUMS'
                 and '__pycache__' not in p.parts and p.suffix!='.pyc')
    assert not any(p.suffix=='.tmp' or p.name.startswith('tmp') for p in paths)
    (PACKAGE/'SHA256SUMS').write_bytes(''.join(sha(p)+'  '+p.relative_to(PACKAGE).as_posix()+'\n' for p in paths).encode('utf-8'))
    paths.append(PACKAGE/'SHA256SUMS')
    with tarfile.open(archive,'w:gz') as tar:
        for p in paths:tar.add(p,arcname=PACKAGE.name+'/'+p.relative_to(PACKAGE).as_posix(),recursive=False)
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers())==len(paths)
        for p in paths:
            assert tar.extractfile(PACKAGE.name+'/'+p.relative_to(PACKAGE).as_posix()).read()==p.read_bytes()
    archive.with_name(archive.name+'.sha256').write_bytes((sha(archive)+'  '+archive.name+'\n').encode('ascii'))
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(paths),sha256=sha(archive)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
