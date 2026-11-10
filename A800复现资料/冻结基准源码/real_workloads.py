"""Fixed-length throughput on real prompts; never used to score answers."""
import hashlib,json,re
from collections import defaultdict
from inventory import rows
SUITES=['longbench','longgenbench','ruler','ruler32','longbenchv2']
def workloads(requested):
    wanted=set(SUITES if 'all' in requested else requested);groups=defaultdict(list)
    for file in ('legacy.jsonl.gz','quality32.jsonl.gz','longbenchv2_slice.jsonl.gz'):
        for r in rows(file):
            suite='ruler32' if file.startswith('quality32') else r['suite']
            if suite not in wanted:continue
            key=(suite,'mixed' if suite=='longbenchv2' else r['task'])
            groups[key].append(r)
    for (suite,task),candidates in sorted(groups.items()):
        selected=[];used=set();hashes=set()
        for r in sorted(candidates,key=lambda r:(-len(r['input_ids']),r['id'])):
            ident=r.get('source_index',r['id'])
            if ident in used or r['input_sha256'] in hashes:continue
            used.add(ident);hashes.add(r['input_sha256']);selected.append(r)
            if len(selected)==8:break
        assert len(selected)==8
        size=min(len(r['input_ids']) for r in selected);prefix=size//2
        bank=[r['input_ids'][:prefix]+r['input_ids'][-(size-prefix):] for r in selected]
        assert len({tuple(x) for x in bank})==8 and size>=1040
        info=[dict(id=r['id'],original_tokens=len(r['input_ids']),removed_middle_tokens=len(r['input_ids'])-size,original_sha256=r['input_sha256']) for r in selected]
        yield f'real_{suite}_{task}_{size}+256',bank,256,info
