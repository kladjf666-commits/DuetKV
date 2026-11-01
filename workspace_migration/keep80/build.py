"""Build an offline overlay on the exact channel-contiguous release used on the 4090."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT/'PhaseKV与Full消融'


def main():
    old = RESULTS/'PhaseKV_4090_channel_only_result'
    reference_protocol = json.loads((old/'protocol.json').read_text())
    target = RESULTS/'PhaseKV_4090_keep80_full.zip'
    files = {}
    with zipfile.ZipFile(RESULTS/'PhaseKV_4090_channel_only.zip') as source:
        for name, expected in reference_protocol['files'].items():
            if name.startswith('channel_impl/') or name in ('retest.py','probe.py','trace_budget.py','batch_attention.py'):
                content = source.read('migration/channel_only/'+name)
                if hashlib.sha256(content).hexdigest() != expected:
                    raise RuntimeError(f'Old release differs from the measured release: {name}')
                files[name] = content
    files['channel_impl/kernels/segmented_plan.py'] = (ROOT/'PhaseKV/kernels/segmented_plan.py').read_bytes()
    for name in ('campaign.py','compare.py','run.sh'):
        files[name] = (HERE/name).read_bytes()
    files = {k:v.replace(b'\r\n',b'\n') for k,v in files.items()}
    native = ROOT/'不同算法的不同数据集测试/native_llama_20261029'
    fields = ('id','suite','task','input_sha256','repeat','status','output_tokens','metric','score',
              'decode_tokens_per_s','indexed_accuracy_diagnostic')
    refs = {}

    def add(label, r, gpu):
        group = refs.setdefault(label, {'gpu': gpu, 'rows': []})
        if group['gpu'] != gpu:
            raise ValueError(f'Historical GPU mismatch for {label}')
        group['rows'].append({k:r[k] for k in fields if k in r})

    gpu = json.loads((native/'gpu_initial.json').read_text())['csv'].split(',')[:2]
    for line in (native/'records.jsonl').open(encoding='utf-8'):
        r = json.loads(line)
        if r['method'] in ('phasekv','full'):
            add('native_4090D_'+r['method'],r,gpu)
    with tarfile.open(RESULTS/'rtx4090_partial_results.tar.gz') as archive:
        for member in archive:
            if '/records/full_fa2/' in member.name and member.isfile() and member.name.endswith('.json'):
                r = json.load(archive.extractfile(member))
                add('rtx4090_full_fa2',r,r['gpu_before']['csv'].split(',')[:2])
    probe = json.loads((old/'summary.json').read_text())['points']['1']
    sample_id = reference_protocol['prompts'][0]
    orig = next(r for r in refs['native_4090D_phasekv']['rows'] if r['id']==sample_id)
    history = dict(groups=refs, frozen=reference_protocol['data'], optimized100_probe=dict(
        gpu=reference_protocol['gpu'], id=sample_id, input_sha256=orig['input_sha256'],
        output_tokens=256, repeats=probe['successful_repeats'], median_tps=probe['total_tokens_per_s']))
    files['history.json.gz'] = gzip.compress(json.dumps(history,separators=(',',':')).encode(),mtime=0)
    files['files.json'] = json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in files.items()},indent=2).encode()
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for k,v in files.items():
            z.writestr('migration/keep80/'+k, v)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    print(target, target.stat().st_size)


if __name__=='__main__':
    main()
