"""Read-only integrity checks, plus small audit reports; never deletes files."""
from pathlib import Path
from collections import Counter
from statistics import median
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
REVIEW = Path(__file__).resolve().parent
STAGE = REVIEW / '拟保留'
PHASE = sys.argv[1] if len(sys.argv) > 1 else 'before'
assert PHASE in ('before', 'after', 'final')
assert str(ROOT) == r'C:\Users\kladjf\Desktop\PhaseKV'
read = lambda p: json.loads(p.read_text('utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = read(REVIEW / '保留来源清单.json')
deletions = read(REVIEW / '拟删除文件清单.json')
inventory = read(REVIEW / '目录盘点.json')
plan = read(REVIEW / '清理方案.json')
assert plan['authorized'] is True and plan['bettermemory_speed_retained'] is True
assert deletions['targets'] == plan['actual_delete_targets']

def safe(relative):
    p = ROOT / relative
    assert p.resolve().is_relative_to(ROOT) and p.resolve() != ROOT, relative
    for part in [p, *p.parents]:
        if part == ROOT:
            break
        if part.exists():
            assert not (part.lstat().st_file_attributes & 0x400), part
    return p

def scan(folder):
    output = {}
    pending = [folder]
    while pending:
        p = pending.pop()
        st = p.lstat()
        assert not (st.st_file_attributes & 0x400), p
        if p.is_dir():
            pending.extend(p.iterdir())
        else:
            output[p.relative_to(ROOT).as_posix()] = {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
    return output

base = STAGE if PHASE == 'before' else ROOT
for name, digest in manifest['staged_sha256'].items():
    assert sha(base / name) == digest, name
if PHASE == 'before':
    assert {p.relative_to(STAGE).as_posix() for p in STAGE.rglob('*') if p.is_file()} == set(manifest['staged_sha256'])
    for row in manifest['copies']:
        source = safe(row['source'])
        assert source.stat().st_size == row['bytes'], row['source']
        assert sha(source) == row['sha256'] == sha(STAGE / row['target']), row['source']
    actual = {}
    for relative in deletions['targets']:
        for name in manifest['staged_sha256']:
            assert name != relative and not name.startswith(relative + '/'), (relative, name)
        assert relative.split('/')[0] not in plan['protected'], relative
        actual.update(scan(safe(relative)))
    expected = {x['path']: {k:x[k] for k in ('bytes', 'mtime_ns')} for x in deletions['files']}
    assert actual == expected, {'new': sorted(actual.keys()-expected.keys())[:10],
        'missing': sorted(expected.keys()-actual.keys())[:10],
        'changed': [k for k in actual.keys() & expected.keys() if actual[k] != expected[k]][:10]}
    original = {x['path']: x for x in inventory['files']}
    for name in manifest['staged_sha256']:
        p = safe(name)
        if p.is_file():
            assert name in original, ('Unexpected existing destination', name)
            st = p.stat()
            assert st.st_size == original[name]['bytes'] and st.st_mtime_ns == original[name]['mtime_ns'], name
else:
    for relative in deletions['targets']:
        assert not safe(relative).exists(), relative

result_root = base / 'A800最终测试'
p90 = result_root / '优化前_Prompt90'
quality = [read(p) for p in (p90 / '质量原始记录').glob('*.json')]
assert len(quality) == 432
assert Counter(r['suite'] for r in quality) == {'longbench':320, 'ruler':64, 'longgenbench':48}
assert len(list((p90 / '速度原始记录/speed').glob('*.json'))) == 16
assert len(list((p90 / '速度原始记录/speed_runs').glob('*.json'))) == 48
for method, counts in plan['quality_counts'].items():
    rows = [read(p) for p in (result_root / '多算法最终对照' / method / '质量原始记录').rglob('*.json')]
    assert dict(Counter(x['status'] for x in rows)) == counts, method
    speeds = [read(p) for p in (result_root / '多算法最终对照' / method / '速度原始记录').glob('*.json')]
    actual_counts = dict(Counter(x['status'] for x in speeds))
    assert actual_counts == {k:v for k,v in plan['speed_counts'][method].items() if k != 'not_tested'}, method

bm = result_root / 'BetterMemory_速度与峰值显存'
points = read(bm / '速度与峰值汇总.json')['speed']
assert len(points) == 16
assert len(list((bm / '逐轮记录').glob('*.json'))) == 48
assert len(list((bm / '预热记录').glob('*.json'))) == 16
raw = ROOT / 'A800最终测试/原始结果/BetterMemory最终评估_20261107/results'
for key, point in points.items():
    rows = [read(bm / f'逐轮记录/{key}_r{i}.json') for i in (1,2,3)]
    assert all(r['status'] == 'ok' for r in rows)
    for metric, field in (('tps','aggregate_tokens_per_s'), ('sequence_tps','per_sequence_tokens_per_s'), ('ttft','ttft_s')):
        assert point[metric] == median(r[field] for r in rows), (key,metric)
    assert point['e2e_tps'] == median(r['batch']*r['output_tokens_per_row']/(r['ttft_s']+r['decode_s']) for r in rows)
    for metric, field in (('prefill_peak_gib','prefill_peak_bytes'), ('decode_peak_gib','decode_peak_bytes'), ('peak_gib','peak_allocated_bytes')):
        assert point[metric] == max(r[field] for r in rows)/2**30, (key,metric)
    if PHASE == 'before':
        for i, row in enumerate(rows, 1):
            source = raw / f'speed/{key}/optimized/r{i}.json'
            original = read(source)
            assert row['source_record_sha256'] == sha(source)
            assert all(original[k] == v for k,v in row.items() if k not in ('source_record_sha256','record_kind'))
for folder in ('逐轮记录', '预热记录'):
    for p in (bm / folder).glob('*.json'):
        r = read(p)
        assert not {'output_ids','output_text','final_kv_bytes','kv_gib'}.intersection(r), p
        if PHASE == 'before' and folder == '预热记录':
            n,t,b = r['case']
            candidates = list((raw / 'warmups' / f'{n}+{t}_B{b}').glob('*/*.json'))
            assert len(candidates) == 1, candidates
            original = read(candidates[0])
            assert r['source_record_sha256'] == sha(candidates[0])
            assert all(original[k] == v for k,v in r.items() if k not in ('source_record_sha256','record_kind'))
gates = read(bm / '实验身份/gates.json')
assert gates['status'] == 'ok' and {x['batch'] for x in gates['checks']} == {1,2,4,8}
assert read(bm / '实验身份/state.json')['phase'] == 'complete'
for name in ('README.md','A800_HANDOFF.md','A800最终测试/阅读说明.txt','A800复现资料/阅读说明.txt',
             'Code-Original-NoChanging - A800 - BetterMemory/README.md'):
    contents = (base / name).read_text('utf-8')
    assert all(x not in contents for x in ('BetterMemory_仅峰值显存','本轮仅保留峰值','本轮只保留峰值','速度已按用户要求不保留')), name

protected = {}
for name in ('.git','.venv','weights','config.txt','.gitignore'):
    protected.update(scan(safe(name)))
protected_hashes = {name:sha(ROOT/name) for name in ('config.txt','.gitignore')}
if PHASE == 'before':
    (REVIEW / '保护内容校验.json').write_text(json.dumps({'files':protected,'sha256':protected_hashes},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
else:
    snapshot = read(REVIEW / '保护内容校验.json')
    assert snapshot['files'] == protected, 'Protected file metadata changed'
    assert snapshot['sha256'] == protected_hashes, 'Protected config content changed'

report = dict(phase=PHASE, verified_kept_files=len(manifest['staged_sha256']), verified_source_copies=len(manifest['copies']),
    planned_removed_files=len(deletions['files']), planned_removed_bytes=sum(r['bytes'] for r in deletions['files']),
    protected_files=len(protected), p90_quality=432, p90_speed_points=16, p90_formal_rounds=48,
    bettermemory_speed_points=16, bettermemory_formal_rounds=48, bettermemory_warmups=16,
    baseline_methods=len(plan['quality_counts']), baseline_quality_records=432*len(plan['quality_counts']), checks='passed')
if PHASE == 'final':
    assert not STAGE.exists()
    final = scan(ROOT)
    report.update(original_workspace_bytes=inventory['total_bytes'],final_workspace_bytes=sum(v['bytes'] for v in final.values()),
        final_workspace_files=len(final),final_root_entries=sorted(p.name for p in ROOT.iterdir()))
    report['net_reclaimed_bytes'] = report['original_workspace_bytes'] - report['final_workspace_bytes']
(REVIEW / f'校验_{PHASE}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
