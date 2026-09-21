"""Download selected public evidence volumes with SHA-256 validation."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--group',default='results');a.add_argument('--extract',action='store_true');args=a.parse_args()
    record=json.loads((ROOT/'release_assets.json').read_text());out=ROOT/'downloads';out.mkdir(exist_ok=True)
    selected=[x for x in record['assets'] if x['name'].startswith(args.group+'-')]
    if not selected:raise SystemExit('No matching asset group')
    for item in selected:
        p=out/item['name']
        if not p.exists() or sha(p)!=item['sha256']:
            request=urllib.request.Request('https://github.com/xuchengfei666/ROBird-Bench/releases/download/'+record['tag']+'/'+item['name'],headers={'User-Agent':'ROBird-reproduction'})
            tmp=p.with_suffix('.part')
            with urllib.request.urlopen(request,timeout=120) as response,tmp.open('wb') as f:
                while True:
                    chunk=response.read(2**20)
                    if not chunk:break
                    f.write(chunk)
            if sha(tmp)!=item['sha256']:raise RuntimeError('Downloaded bytes do not match: '+item['name'])
            tmp.replace(p)
        if args.extract:
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    target=(ROOT/name).resolve()
                    if ROOT not in target.parents:raise RuntimeError('Unsafe archive path')
                z.extractall(ROOT)
        print('VERIFIED',item['name'])
if __name__=='__main__':main()
