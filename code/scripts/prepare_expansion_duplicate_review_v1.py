"""Hash-bound contact sheets for outcome-blinded AI visual audit; never grants PASS."""
from pathlib import Path
import sys
import json
from collections import Counter
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/src'))
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
from robird.io import atomic_write_json,atomic_write_csv,sha256_file
from robird.rsos_suite_v1 import atomic_binary
from robird.external_expansion_v2 import verify_files
from robird.external_cohort_v1_1 import image_signature,GateStop
from robird.artifact_tree_v1 import verify_tree

SOURCE=ROOT/'code/results/external_expansion_v2_1'
REPORT=ROOT/'code/results/external_expansion_duplicate_review_v1'
DATA=Path('E:/Datasets/ROBird-Bench/external_expansion_v2_1')
OUT=Path('E:/Datasets/ROBird-Bench/external_expansion_duplicate_review_v1')

def identity(value):
    if value is None or str(value).strip() in ('','nan'):
        return None
    number=Decimal(str(value))
    if not number.is_finite() or number!=number.to_integral_value():
        raise ValueError('Nonintegral source ID')
    return int(number)

def main():
    if (REPORT/'index.json').exists():
        raise FileExistsError('Review already prepared; use existing bound sheets')
    freeze=ROOT/'FROZEN_EXTERNAL_EXPANSION_V2_1.json'
    if sha256_file(freeze)!='6e5cecd4680aa8869cd4aa61d6788c9c1848baa567536f491d6dbde3f1e96112':
        raise GateStop('Parent freeze changed')
    verify_files(json.loads(freeze.read_text(encoding='utf-8'))['files'])
    for path in [SOURCE/'metadata_done.json',SOURCE/'download_done.json',DATA/'reference_signatures_done.json']:
        verify_tree(path)
    frame=pd.read_csv(SOURCE/'duplicate_review.csv',keep_default_na=False,dtype=str)
    downloaded=pd.read_csv(SOURCE/'downloaded.csv',keep_default_na=False)
    references=pd.read_csv(DATA/'reference_signatures.csv',keep_default_na=False,dtype=str)
    expected={str(Path(r.local_path).resolve()):r.sha256 for r in references.itertuples()}
    expected.update({str(Path(r.local_path).resolve()):r.sha256 for r in downloaded.itertuples()})
    # All selected download ledgers and all historical reference bytes, not only flagged photos.
    for path in sorted((DATA/'download_ledger').glob('*.json')):
        verify_tree(path)
    for path,digest in expected.items():
        if sha256_file(Path(path))!=digest:
            raise GateStop('Byte drift: '+path)
    print('All selected and reference bytes verified',flush=True)
    rows=[]; signatures={}
    for i,row in enumerate(frame.to_dict('records')):
        item=dict(pair_id=i,**row)
        for key in ('photo_id','observation_id','reference_photo_id','reference_observation_id'):
            item[key]=identity(row[key])
        for key in ('path_a','path_b'):
            path=str(Path(row[key]).resolve())
            if path not in expected:raise GateStop('Unknown image path')
            if path not in signatures:signatures[path]=image_signature(Path(path))
            sig=signatures[path]
            if sig['sha256']!=expected[path]:raise GateStop('Image changed')
            item[key]=path;item[key+'_sha256']=sig['sha256']
            item[key+'_width']=sig['width'];item[key+'_height']=sig['height']
        a,b=[signatures[item[k]] for k in ('path_a','path_b')]
        exact=a['sha256']==b['sha256'];dist=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
        if exact!=(row['exact_bytes'].lower()=='true') or dist!=int(row['dhash_distance']):
            raise GateStop('Candidate signature drift')
        item['exact_bytes']=exact;item['dhash_distance']=dist
        item['same_observation']=item['observation_id']==item['reference_observation_id']
        item['sheet']=str(OUT/f'pairs_{i//12:02d}.jpg')
        rows.append(item)
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    sheets={}
    for start in range(0,len(rows),12):
        canvas=Image.new('RGB',(1320,1536),'#e5e7eb');draw=ImageDraw.Draw(canvas)
        for slot,item in enumerate(rows[start:start+12]):
            x=(slot%2)*660;y=(slot//2)*256
            label=f"P{item['pair_id']:03d} {item['source']} d={item['dhash_distance']} exact={int(item['exact_bytes'])} sameObs={int(item['same_observation'])}"
            draw.text((x+5,y+3),label,font=font,fill='black')
            for side,key in enumerate(('path_a','path_b')):
                with Image.open(item[key]) as img:
                    image=ImageOps.contain(img.convert('RGB'),(322,220))
                canvas.paste(image,(x+side*330+4+(322-image.width)//2,y+30+(220-image.height)//2))
        path=OUT/f'pairs_{start//12:02d}.jpg'
        if path.exists():raise FileExistsError(path)
        atomic_binary(path,lambda f:canvas.save(f,format='JPEG',quality=94))
        sheets[str(path)]=sha256_file(path)
    atomic_write_json(REPORT/'index.json',dict(source_csv=str(SOURCE/'duplicate_review.csv'),
        source_sha256=sha256_file(SOURCE/'duplicate_review.csv'),parent_freeze_sha256=sha256_file(freeze),
        reviewer_type='AI_ASSISTED_VISUAL_REVIEW',independent_human_adjudication=False,
        outcome_blinded=True,pairs=rows,sheets=sheets),refuse_if_exists=True)
    atomic_write_csv(REPORT/'pair_inventory.csv',pd.DataFrame(rows),refuse_if_exists=True)
    summary=dict(pairs=len(rows),sources=dict(Counter(r['source'] for r in rows)),
        same_observation=sum(r['same_observation'] for r in rows),exact_pairs=sum(r['exact_bytes'] for r in rows),
        selected_images=len(downloaded),all_reference_paths=len(expected),review_images=len(signatures),
        sheets=len(sheets),original_g6_pass=False,status='PREPARED_NOT_ADJUDICATED',
        index_sha256=sha256_file(REPORT/'index.json'))
    atomic_write_json(REPORT/'preparation_audit.json',summary,refuse_if_exists=True)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
