"""Build the optional offline torch/CUDA supplement for the existing CP310 overlay."""
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import shutil
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, unquote
import json
from pathlib import Path
import zipfile
import requests

from build_overlay import CACHE, ROOT, HERE, DELIVERY, sha

TORCH_NAME = 'torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl'
TORCH_HASH = '92af92c569de5da937dd1afb45ecfdd598ec1254cf2e49e3d698cb24d71aae14'
INDEX = 'https://download.pytorch.org/whl/cu121/torch/'


def download_large(path, url, size, digest):
    if path.is_file() and sha(path)==digest:
        return path,digest
    cache = CACHE/(path.name+'.chunks')
    cache.mkdir(exist_ok=True)
    chunk = 4<<20
    def get_part(start):
        end = min(start+chunk,size)-1
        p = cache/str(start)
        if p.is_file() and p.stat().st_size==end-start+1: return p
        for attempt in range(3):
            try:
                r = requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,65))
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}' or len(r.content)!=end-start+1:
                    raise ValueError(f'Invalid byte range response: {path.name}/{start}: {r.status_code}')
                tmp=p.with_suffix('.tmp')
                tmp.write_bytes(r.content)
                tmp.replace(p)
                return p
            except (requests.RequestException, ValueError):
                if attempt==2: raise
                sleep(attempt+1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts=list(pool.map(get_part,range(0,size,chunk)))
    assembled = cache/'assembled'
    with assembled.open('wb') as out:
        for part in parts:
            with part.open('rb') as f: shutil.copyfileobj(f,out)
    if sha(assembled)!=digest:
        raise RuntimeError(f'Wheel hash mismatch: {path.name}')
    assembled.replace(path)
    for part in parts: part.unlink()
    print('verified',path.name,flush=True)
    return path,digest


def cuda_wheel(pin):
    name,ver=pin.split('==')
    response=requests.get(f'https://pypi.org/pypi/{name}/{ver}/json',timeout=30)
    response.raise_for_status()
    entry,=[e for e in response.json()['urls'] if e['filename'].endswith('x86_64.whl') and 'manylinux' in e['filename']]
    return download_large(CACHE/entry['filename'],entry['url'],entry['size'],entry['digests']['sha256'])


def torch_wheel():
    path = CACHE/TORCH_NAME
    if path.exists() and sha(path)==TORCH_HASH:
        return path, TORCH_HASH
    class Links(HTMLParser):
        links = []
        def handle_starttag(self, tag, attrs):
            if tag == 'a': self.links.append(dict(attrs).get('href',''))
    links = Links()
    response = requests.get(INDEX, timeout=45)
    response.raise_for_status()
    links.feed(response.text)
    url, = [urljoin(INDEX,u) for u in links.links
            if unquote(urlsplit(u).path).split('/')[-1]==TORCH_NAME and urlsplit(u).fragment=='sha256='+TORCH_HASH]
    return download_large(path,url.split('#')[0],780429430,TORCH_HASH)


def main():
    CACHE.mkdir(exist_ok=True)
    frozen = next(ROOT.glob('*/native_llama_20261029/pip-freeze.txt')).read_text()
    pins = [s for s in frozen.splitlines() if s.startswith('nvidia-') and '==' in s]
    if len(pins)!=12: raise RuntimeError('Unexpected base CUDA stack')
    with ThreadPoolExecutor(max_workers=8) as pool:
        torch_job = pool.submit(torch_wheel)
        table = dict(pool.map(cuda_wheel, pins))
        path, digest = torch_job.result()
        table[path] = digest
    target = DELIVERY/'PhaseKV_torch_cuda121_offline.zip'
    partial = target.with_suffix('.partial')
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in sorted(table):
            z.write(path,'migration/runtime_wheels/'+path.name)
        z.writestr('migration/runtime_manifest.json',json.dumps({p.name:h for p,h in sorted(table.items())},indent=2))
        z.write(HERE/'install.py','migration/install.py')
        z.write(HERE/'warm_sm80.py','migration/warm_sm80.py')
        z.write(HERE/'parallel.py','migration/first_8k.py')
        z.write(HERE/'OFFLINE_A800.txt','migration/OFFLINE_A800.txt')
    with zipfile.ZipFile(partial) as z:
        if z.testzip(): raise RuntimeError('Corrupt supplement ZIP')
    partial.replace(target)
    target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n',encoding='ascii')
    print(target,target.stat().st_size,sha(target),flush=True)


if __name__ == '__main__':
    main()
