"""Create seeded Sudoku instances and synthetic 9x9 MNIST mosaics."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from sudoku_generator import make_puzzle
from torchvision.datasets import MNIST

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="data/processed/puzzles"); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--count",type=int,default=100); ap.add_argument("--givens",type=int,default=30)
    ap.add_argument("--misreads",type=int,default=2)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    mn=MNIST(str(out.parent/"raw"),train=True,download=True)
    images=mn.data.numpy(); labels=mn.targets.numpy()
    by_digit={d: images[labels==d] for d in range(1,10)}
    rng=np.random.default_rng(a.seed); records=[]
    for i in range(a.count):
        sol,puz=make_puzzle(a.seed+i,a.givens,True)
        read=puz.copy(); injected=[]
        blank=[(r,c) for r in range(9) for c in range(9) if puz[r,c]==0]
        # Inject only into blank cells; this creates realistic-looking but conflicting readings.
        rng.shuffle(blank)
        for r,c in blank:
            true=int(sol[r,c])
            peers=set(int(x) for x in read[r,:] if x) | set(int(x) for x in read[:,c] if x)
            br,bc=(r//3)*3,(c//3)*3
            peers |= set(int(x) for x in read[br:br+3,bc:bc+3].ravel() if x)
            choices=sorted(d for d in peers if d!=true)
            if not choices:
                continue
            wrong=int(rng.choice(choices))
            read[r,c]=wrong
            injected.append({"row":r,"col":c,"true_digit":true,"read_digit":wrong})
            if len(injected) == a.misreads:
                break
        if len(injected) != a.misreads:
            raise RuntimeError(f"could inject only {len(injected)} of {a.misreads} requested conflicting errors")
        canvas=Image.new("L",(252,252),0); draw=ImageDraw.Draw(canvas)
        injected_map={(item["row"],item["col"]):item for item in injected}
        for r in range(9):
            for c in range(9):
                d=int(read[r,c])
                if d:
                    if (r,c) in injected_map:
                        # A hard cell is a true-class MNIST glyph blended with a
                        # wrong-class distractor. The ground-truth digit remains
                        # `true_digit`; the injected read is `read_digit`.
                        item=injected_map[(r,c)]
                        true_img=Image.fromarray(by_digit[item["true_digit"]][rng.integers(len(by_digit[item["true_digit"]]))])
                        distractor=Image.fromarray(by_digit[item["read_digit"]][rng.integers(len(by_digit[item["read_digit"]]))])
                        im=Image.blend(true_img,distractor,alpha=0.45).resize((24,24))
                    else:
                        im=Image.fromarray(by_digit[d][rng.integers(len(by_digit[d]))]).resize((24,24))
                    canvas.paste(im,(c*28+2,r*28+2))
                if c%3==0: draw.line((c*28,0,c*28,252),fill=96,width=1)
                if r%3==0: draw.line((0,r*28,252,r*28),fill=96,width=1)
        for k in range(10):
            draw.line((k*28,0,k*28,252),fill=255 if k%3==0 else 160,width=2 if k%3==0 else 1)
            draw.line((0,k*28,252,k*28),fill=255 if k%3==0 else 160,width=2 if k%3==0 else 1)
        mp=out/f"mosaic_{i:04d}.png"; canvas.save(mp)
        records.append({"seed":a.seed+i,"solution":sol.tolist(),"puzzle":puz.tolist(),
          "givens":int(np.count_nonzero(puz)),"mosaic_path":mp.name,"injected_misreads":injected,
          "hard_digit_rule":"true MNIST glyph blended with a wrong-class distractor at alpha=0.45",
          "read_grid":read.tolist()})
    with (out/"puzzles.jsonl").open("w",encoding="utf-8") as f:
        for rec in records: f.write(json.dumps(rec)+"\n")
    print(f"Wrote {len(records)} puzzles to {out}")
if __name__=="__main__": main()
