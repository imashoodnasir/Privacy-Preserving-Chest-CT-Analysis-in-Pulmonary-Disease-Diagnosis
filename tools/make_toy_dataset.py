import os
import argparse
import numpy as np
from pathlib import Path

def make_volume(D=32, H=128, W=128, pattern=0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    vol = rng.normal(0, 1, size=(D, H, W)).astype(np.float32)
    # inject a class-dependent synthetic pattern (non-medical, just to test code)
    if pattern == 0:
        # central blob
        zz, yy, xx = np.ogrid[:D, :H, :W]
        mask = (yy - H/2)**2 + (xx - W/2)**2 < (min(H,W)/6)**2
        vol[:, mask] += 2.0
    elif pattern == 1:
        # diagonal stripe
        for z in range(D):
            for i in range(min(H,W)):
                vol[z, i, i] += 3.0
    else:
        # ring
        zz, yy, xx = np.ogrid[:D, :H, :W]
        r = np.sqrt((yy - H/2)**2 + (xx - W/2)**2)
        ring = (r > min(H,W)/5) & (r < min(H,W)/4)
        vol[:, ring] += 2.5
    # clip + standardize like HU preprocessing placeholder
    vol = np.clip(vol, -4, 6)
    return vol

def save_case(out_dir, idx, vol, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"case_{idx:04d}.npz", volume=vol.astype(np.float32), label=int(label))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--train", type=int, default=120)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)

    for c in range(1, args.clients+1):
        client = base / f"client_{c:02d}"
        # simulate site shift: different noise scale per client
        site_scale = 1.0 + (c-1) * 0.25

        for split, n in [("train", args.train), ("val", args.val), ("test", args.test)]:
            for i in range(n):
                label = int(rng.integers(0, args.classes))
                vol = make_volume(pattern=label, rng=rng) * site_scale
                save_case(client / split, i, vol, label)

    print(f"Toy dataset created at: {base.resolve()} with {args.clients} clients.")

if __name__ == "__main__":
    main()
