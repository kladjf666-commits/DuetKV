"""Exclusive same-window accounting. CPU labels indicate coincidence, not causality."""
from collections import Counter, defaultdict

LABEL = 'probe/window'
GPU = {'kernel', 'gpu_memcpy', 'gpu_memset'}
HOST = ['sync API', 'launch API', 'other CUDA API', 'cache finish', 'cache append', 'workspace',
        'attention dispatch', 'argmax', 'PyTorch CPU op', 'model forward', 'other host']


def account(events, steps=16):
    markers = [e for e in events if e.get('name') == LABEL and e.get('ph') == 'X']
    if len(markers) != 1:
        raise ValueError('Missing or ambiguous CPU window marker')
    marker = markers[0]
    start, stop = marker['ts'], marker['ts'] + marker['dur']
    edges = defaultdict(Counter)
    edges[start]
    edges[stop]
    device_sum = 0.
    for e in events:
        if e.get('ph') != 'X' or 'dur' not in e:
            continue
        a, b = max(start, e['ts']), min(stop, e['ts'] + e['dur'])
        if a >= b:
            continue
        name, category = e.get('name', ''), e.get('cat')
        if category in GPU:
            kind = 'GPU active'
            device_sum += b-a
        elif (e.get('pid'), e.get('tid')) != (marker['pid'], marker['tid']):
            continue
        elif category in ('cuda_runtime', 'cuda_driver'):
            kind = ('sync API' if 'Synchronize' in name else
                    'launch API' if 'Launch' in name else 'other CUDA API')
        elif name.startswith('probe/') and name[6:] in HOST:
            kind = name[6:]
        elif category == 'cpu_op':
            kind = 'PyTorch CPU op'
        else:
            continue
        edges[a][kind] += 1
        edges[b][kind] -= 1
    active, totals = Counter(), Counter()
    previous = start
    for stamp, edits in sorted(edges.items()):
        if stamp > previous:
            kind = next((k for k in ['GPU active', *HOST] if active[k]), 'other host')
            totals[kind] += stamp-previous
        active.update(edits)
        previous = stamp
    if totals['GPU active'] <= 0:
        raise RuntimeError('No CUPTI device activities inside window')
    scale = 1000*steps  # trace timestamps in microseconds
    return {'wall_ms':(stop-start)/scale, 'gpu_union_ms':totals['GPU active']/scale,
            'gpu_sum_ms':device_sum/scale,
            'idle_ms':{k:totals[k]/scale for k in HOST}}
