"""Parallel transport for the official essay URL list; same parsing/order as upstream.

Only HTTP transport is modernized (HTTPS and raw.githubusercontent.com). All
download status and payload hashes are recorded; incomplete corpora are rejected.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import html2text
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT/'bench_sources/RULER/scripts/data/synthetic/json'
CACHE = ROOT/'bench_assets/essay_downloads'


def fetch(url):
    name=url.rsplit('/',1)[-1].replace('.html','.txt')
    kind='html' if '.html' in url else 'repo'
    target=CACHE/kind/name
    target.parent.mkdir(parents=True,exist_ok=True)
    transport=url.replace('http://www.paulgraham.com/','https://www.paulgraham.com/')
    transport=transport.replace('https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/',
                                'https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/')
    error=None
    for attempt in range(3):
        try:
            if not target.exists():
                raw=urllib.request.urlopen(transport,timeout=25).read()
                if kind=='html':
                    h=html2text.HTML2Text()
                    h.ignore_images=True; h.ignore_tables=True; h.escape_all=True
                    h.reference_links=False; h.mark_code=False
                    content=h.handle(str(BeautifulSoup(raw.decode('unicode_escape','ignore'),'html.parser').find('font')))
                else:
                    content=raw.decode('utf-8')
                if not content.strip() or content.strip()=='None':
                    raise ValueError('Empty essay')
                target.write_text(content,encoding='utf-8')
            return {'url':url,'transport_url':transport,'file':str(target.relative_to(ROOT)),
                    'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'status':'ok'}
        except Exception as exc:
            error=repr(exc); time.sleep(1)
    return {'url':url,'status':'error','error':error}


if __name__=='__main__':
    urls=(FOLDER/'PaulGrahamEssays_URLs.txt').read_text().splitlines()
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows=[]
        for row in pool.map(fetch,urls):
            rows.append(row)
            print(f'{len(rows)}/{len(urls)} {row["status"]} {row["url"]}',flush=True)
    (ROOT/'bench_assets/essay_sources.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if any(r['status']!='ok' for r in rows):
        raise RuntimeError('Incomplete essay corpus; see essay_sources.json')
    text=''.join(p.read_text('utf-8') for kind in ('repo','html') for p in sorted((CACHE/kind).glob('*.txt')))
    (FOLDER/'PaulGrahamEssays.json').write_text(json.dumps({'text':text}),encoding='utf-8')
