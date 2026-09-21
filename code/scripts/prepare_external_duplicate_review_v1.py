"""Build outcome-blinded pair contact sheets; no automatic adjudication."""
from pathlib import Path
import json
import hashlib
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('E:/Datasets/ROBird-Bench/external_duplicate_review_v1')
SOURCE=ROOT/'code/results/external_cohort_v1_3/duplicate_review.csv'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(SOURCE,keep_default_na=False)
    index=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for start in range(0,len(frame),5):
        sheet=Image.new('RGB',(1000,1450),'#dddddd');draw=ImageDraw.Draw(sheet)
        for slot,(_,row) in enumerate(frame.iloc[start:start+5].iterrows()):
            pair_id=start+slot
            item=dict(pair_id=pair_id,source=row.source,photo_id=int(row.photo_id),
                observation_id=int(row.observation_id),reference_observation_id=str(row.reference_observation_id),
                dhash_distance=int(row.dhash_distance),exact_bytes=str(row.exact_bytes))
            for col,key in enumerate(('path_a','path_b')):
                path=Path(row[key]);img=Image.open(path).convert('RGB')
                item[key]=str(path);item[key+'_sha256']=sha(path);item[key+'_size']=list(img.size)
                image=ImageOps.contain(img,(492,250))
                x=col*500+(500-image.width)//2;y=slot*290+35+(250-image.height)//2
                sheet.paste(image,(x,y))
                label=f"Pair {pair_id:02d} {'A' if col==0 else 'B'}  {path.stem}  {img.width}x{img.height}"
                draw.text((col*500+5,slot*290+5),label,font=font,fill='black')
            index.append(item)
        path=OUT/f'pairs_{start:02d}_{min(start+4,len(frame)-1):02d}.jpg'
        if path.exists():raise FileExistsError(path)
        sheet.save(path,quality=93)
    payload=dict(source_sha256=sha(SOURCE),pairs=index)
    with (OUT/'index.json').open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
    print(json.dumps(dict(pairs=len(frame),sheets=(len(frame)+4)//5,path=str(OUT))))

if __name__=='__main__':main()
