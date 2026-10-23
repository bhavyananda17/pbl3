"""Copy the trained models from the base-model repo into models/base/, without optimizer state.

    python scripts/export_base_models.py --old-repo ../facial-emotion-recognition-cnn

The old repo's .keras files also store the Adam optimizer's moment estimates, which make them
about 3x larger than the weights alone (RAF-DB: 120 MB). This script loads each model with
compile=False, so no optimizer exists, saves it again, reloads the new file and checks that every
weight (trainable and BatchNorm moving statistics) is bit-identical to the original.
Nothing is retrained and the old repo is only read. The sha256 of both files is printed for
configs/base.yaml.
"""

import argparse
import hashlib
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("rafdb_clean", "fer2013_clean")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(src, dst):
    original = keras.models.load_model(src, compile=False)
    original.save(dst)
    reloaded = keras.models.load_model(dst, compile=False)
    a, b = original.get_weights(), reloaded.get_weights()
    assert len(a) == len(b), f"{dst.name}: weight count changed"
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y), f"{dst.name}: weight {i} differs"
    return len(a)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-repo", required=True, help="path to facial-emotion-recognition-cnn")
    args = p.parse_args()
    out = ROOT / "models" / "base"
    out.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        src = Path(args.old_repo) / "models" / f"{name}.keras"
        dst = out / f"{name}.keras"
        n = export(src, dst)
        print(f"{name}: {n} weight arrays identical | {src.stat().st_size / 1e6:.1f} MB -> "
              f"{dst.stat().st_size / 1e6:.1f} MB")
        print(f"  original sha256: {sha256(src)}")
        print(f"  exported sha256: {sha256(dst)}")


if __name__ == "__main__":
    main()
