"""Record real local checks; do not turn unavailable GPU checks into passes."""
from pathlib import Path
import ast
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare import verify


def main():
    target = ROOT / 'validation'
    target.mkdir(exist_ok=True)
    manifest = verify()
    source_files = [*ROOT.glob('*.py'), *(ROOT / 'tests').glob('*.py')]
    for file in source_files:
        ast.parse(file.read_text('utf-8'), filename=str(file))
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8',
           'HF_HOME': str(ROOT / '.cache/local_validation'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    process = subprocess.run([sys.executable, str(ROOT / 'tests/test_native_contract.py')],
                              capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    (target / 'cpu_tests.txt').write_text(process.stdout + process.stderr, encoding='utf-8')
    import torch
    import transformers
    report = {'date': '2026-10-28', 'python': sys.version, 'torch': torch.__version__,
              'transformers': transformers.__version__, 'cuda_available': torch.cuda.is_available(),
              'source_hashes': {'status': 'passed', 'files': len(manifest['files'])},
              'python_syntax': {'status': 'passed', 'files': len(source_files)},
              'cpu_native_contract': {'status': 'passed' if process.returncode == 0 else 'failed',
                                      'log': 'cpu_tests.txt'},
              'native_cuda_compilation': 'not_run_on_this_host',
              'cuda_kernel_numerics': 'not_run_on_this_host',
              'real_llama31_8b_fp16': 'not_run_on_this_host',
              'linux_shell_execution': 'not_run_on_this_windows_host'}
    (target / 'local_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
