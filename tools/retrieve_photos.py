"""Retrieve separately licensed photographs; require exact recorded byte hashes."""
from pathlib import Path
import argparse,csv,hashlib,json,urllib.parse,urllib.request
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.limit is not None and a.limit<1:raise ValueError('limit must be positive')
    a.output.mkdir(parents=True,exist_ok=True);seen=set();count=0;failures=[]
    with a.manifest.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            pid=row['photo_id']
            if pid in seen:continue
            seen.add(pid)
            if a.limit and count>=a.limit:break
            count+=1
            if row['license_code'] not in ('cc0','cc-by','cc-by-sa') or not row['attribution']:raise ValueError('Unapproved or incomplete source rights')
            url=row['url'];parsed=urllib.parse.urlparse(url)
            if parsed.scheme!='https' or not (parsed.hostname.endswith('inaturalist.org') or parsed.hostname.endswith('amazonaws.com')):raise ValueError('Unexpected image host')
            if not pid.isdigit():raise ValueError('Invalid photo identity')
            target=a.output/(pid+'.image')
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Existing bytes differ; not overwritten')
                continue
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ROBird-public-retrieval'}),timeout=60) as r:data=r.read(25*2**20)
                if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Source bytes changed')
                with target.open('xb') as out:out.write(data)
            except Exception as error:failures.append(dict(photo_id=pid,error=type(error).__name__))
    print(json.dumps(dict(checked_or_requested=count,failed=len(failures),failures=failures),indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
