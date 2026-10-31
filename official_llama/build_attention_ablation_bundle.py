"""Build an additive overlay; existing native source/data files are not replaced."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
NAMES = ['.gitattributes', 'attention_ablation.py', 'ablation_runtime.py', 'ablation_attention.py',
         'ablation_kernels.py', 'ablation_generate.py', 'ablation_validate.py',
         'ablation_analysis.py', 'start_attention_ablation.sh', 'ATTENTION_ABLATION.md',
         'verify_attention_ablation_bundle.py', 'build_attention_ablation_bundle.py',
         'tests/test_attention_ablation.py', 'validation/attention_ablation_local.json']
BASE = ['benchmark.py', 'benchmark_generate.py', 'benchmark_metrics.py', 'native_runtime.py',
        'prepare.py', 'config.json', 'sources.json', 'bench_assets/frozen/manifest.json',
        'bench_assets/frozen/samples.jsonl.gz',
        'bench_sources/RULER/scripts/eval/synthetic/constants.py',
        'vendor/SnapKV/experiments/LongBench/metrics.py', 'vendor/SCOPE/eval_longgenbench.py']


def main():
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    validation = json.loads((ROOT/'validation/attention_ablation_local.json').read_text('utf-8'))
    if validation.get('status') != 'passed' or validation.get('gpu_tests_executed') is not False:
        raise RuntimeError('Expected successful local checks with GPU execution explicitly pending')
    manifest = {'experiment': 'attention-cache-2x2-v1', 'baseline_commit': 'e0179c3baf6c6ea5eb047d3a3e209b84afc3ffb9',
                'files': {name: sha(ROOT/name) for name in NAMES},
                'base_files': {name: sha(ROOT/name) for name in BASE},
                'local_checks': 'passed', 'gpu_checks': 'pending_on_server'}
    mp = ROOT/'attention_ablation_bundle.json'
    mp.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    target = ROOT/'delivery/PhaseKV_attention_ablation_20261031.zip'
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in [*NAMES, mp.name]:
            z.write(ROOT/name, 'official_llama/'+name)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(z.read('official_llama/'+name)).hexdigest() == expected
    digest = sha(target)
    target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n',encoding='ascii',newline='\n')
    print(json.dumps({'file': str(target), 'bytes': target.stat().st_size, 'sha256': digest}, indent=2))


if __name__ == '__main__':
    main()
