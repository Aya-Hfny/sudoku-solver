"""Download MNIST, keep digits 1-9, and make deterministic 50k/10k + test split."""
import argparse, json
from pathlib import Path
import numpy as np
from torchvision.datasets import MNIST

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds_train = MNIST(str(out / "raw"), train=True, download=True)
    ds_test = MNIST(str(out / "raw"), train=False, download=True)
    x_all = ds_train.data.numpy().astype("float32") / 255.0
    y_all = ds_train.targets.numpy().astype("int64")
    rng = np.random.default_rng(args.seed)
    # Split the original 60,000 records first (50k/10k), then drop zero
    # independently from each partition as required by the 1-9 classifier.
    idx = rng.permutation(len(y_all))
    tr_raw, va_raw = idx[:50000], idx[50000:]
    tr_keep, va_keep = y_all[tr_raw] != 0, y_all[va_raw] != 0
    x_train, y_train = x_all[tr_raw][tr_keep], y_all[tr_raw][tr_keep]
    x_val, y_val = x_all[va_raw][va_keep], y_all[va_raw][va_keep]
    xt = ds_test.data.numpy().astype("float32") / 255.0
    yt = ds_test.targets.numpy().astype("int64")
    kt = yt != 0
    np.savez_compressed(out / "mnist_1_9.npz",
        x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
        x_test=xt[kt], y_test=yt[kt])
    dictionary = {
      "seed": args.seed,
      "source": "MNIST",
      "filter": "labels 1-9; label 0 excluded",
      "normalization": "float32 pixels divided by 255",
      "split": "seeded permutation of original 60,000 official training records: first 50,000 train, last 10,000 validation; then digit 0 is removed separately from each partition; official test untouched",
      "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k,v in
        {"x_train":x_train, "y_train":y_train, "x_val":x_val, "y_val":y_val, "x_test":xt[kt], "y_test":yt[kt]}.items()}
    }
    (out / "data_dictionary.json").write_text(json.dumps(dictionary, indent=2), encoding="utf-8")
    print(f"Saved {out/'mnist_1_9.npz'}")
if __name__ == "__main__": main()
