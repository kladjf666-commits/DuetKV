"""Small CPU-only helpers for the isolated BetterMemory A/B benchmark."""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib, json, os, time

ROOT = Path(__file__).resolve().parents[1]
BASE = '180f71ceff70ed6a046bd8ae7ad1f521181c8160d81d5d455194b04dfa539ef2'
PROFILE = dict(id='J_p90_w128_o32', prefill_keep_ratio=.9, decode_keep_ratio=.8,
               prompt_bits=4, cold_bits=4, recent_window=128, obs_window=32)
OPT = dict(fused_quantize=True, reuse_workspace=True, paged_cold=True, page_slots=32,
           cold_group=8, specialized_4bit=True, gqa_reuse=True, hot_ring=True,
           channel_contiguous_k=True, a800_prompt_mma=True)
CASES = [(8192, 256, 1), (32768, 256, 8)]
ARMS = ('baseline', 'optimized')
_timed_depth = 0

@contextmanager
def timed_region():
    """Forbid this benchmark's checkpoint/status writes during measurement."""
    global _timed_depth
    _timed_depth += 1
    try:
        yield
    finally:
        _timed_depth -= 1

def options(p):
    return dict({k: v for k, v in p.items() if k != 'id'}, group_size=32,
                lazy_g=16, sink_tokens=0, optimizations=dict(OPT))

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write(path, value, *, new=False):
    if _timed_depth:
        raise RuntimeError('Checkpoint/status write attempted inside measured region')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        with temp.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        if new:
            if os.name == 'nt': os.rename(temp, path)
            else: os.link(temp, path)
        else: os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def verify(root, expected=None):
    root = Path(root).resolve(); manifest = sha(root / 'SHA256SUMS')
    if expected and manifest != expected: raise ValueError('Manifest changed: ' + str(root))
    for line in (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        checksum, name = line.split('  ', 1); path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != checksum:
            raise ValueError('Source changed: ' + str(path))
    return manifest

@contextmanager
def lock(path):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def case_name(case):
    n, t, b = case
    return f'{n}+{t}_B{b}'

def expected_paths(plan):
    return ([f'pairs/{case_name(case)}/r{r}.json' for case in CASES for r in (1, 2, 3)]
            + [f'diagnostics/{case_name(case)}.json' for case in CASES])

def validate_saved(record, experiment):
    if record['experiment'] != experiment: raise ValueError('Result identity changed')
    if record['status'] != 'ok': raise RuntimeError('Saved failure requires review; no automatic retry')
    return record

def arm_order(case_index, round_index):
    return ARMS if (case_index + round_index) % 2 else ARMS[::-1]
