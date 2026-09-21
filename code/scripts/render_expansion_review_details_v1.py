"""Additional full-image and central crops; displays only, never automatic decisions."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/src'))
from PIL import Image,ImageOps,ImageDraw,ImageFont
from robird.io import atomic_write_json,sha256_file
from robird.rsos_suite_v1 import atomic_binary
REPORT=ROOT/'code/results/external_expansion_duplicate_review_v1'
OUT=Path('E:/Datasets/ROBird-Bench/external_expansion_duplicate_review_v1/details')

def main():
    index=json.loads((REPORT/'index.json').read_text(encoding='utf-8'))
    notes=json.loads((REPORT/'first_visual_pass.json').read_text(encoding='utf-8'))
    ids=[r['pair_id'] for r in notes['pairs'] if r['decision'] in ('UNRESOLVED','NEAR_DUPLICATE_WITHIN_OBSERVATION')]
    rows=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for i in ids:
        r=index['pairs'][i];canvas=Image.new('RGB',(1400,1050),'#dddddd');draw=ImageDraw.Draw(canvas)
        draw.text((8,4),f'Pair {i}: top full image; bottom central 40% crop (display zoom only)',font=font,fill='black')
        for side,key in enumerate(('path_a','path_b')):
            if sha256_file(Path(r[key]))!=r[key+'_sha256']:raise ValueError('Image changed')
            with Image.open(r[key]) as im:img=im.convert('RGB')
            full=ImageOps.contain(img,(684,465));x=side*700
            draw.text((x+8,31),f'{key}: {Path(r[key]).name} {img.size}',font=font,fill='black')
            canvas.paste(full,(x+(700-full.width)//2,60+(465-full.height)//2))
            w,h=img.size;box=(int(w*.3),int(h*.3),int(w*.7),int(h*.7))
            crop=ImageOps.contain(img.crop(box),(684,480))
            canvas.paste(crop,(x+(700-crop.width)//2,555+(480-crop.height)//2))
            draw.text((x+8,530),f'Central crop original coords: {box}',font=font,fill='black')
        path=OUT/f'pair_{i:03d}.jpg'
        if path.exists():raise FileExistsError(path)
        atomic_binary(path,lambda f:canvas.save(f,format='JPEG',quality=96))
        rows.append(dict(pair_id=i,path=str(path),sha256=sha256_file(path)))
    atomic_write_json(REPORT/'detail_index.json',dict(first_pass_sha256=sha256_file(REPORT/'first_visual_pass.json'),
        panels=rows,central_crop_fraction=.4),refuse_if_exists=True)
    print(json.dumps(dict(pairs=len(rows),ids=ids)),flush=True)

if __name__=='__main__':main()
