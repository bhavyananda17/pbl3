"""Check that the frozen model here gives the same predictions as the old repo.

    python scripts/check_match.py --old-repo ../facial-emotion-recognition-cnn

Uses 50 RAF-DB test images (seeded sample of the old repo's results/rafdb/clean/predictions.csv),
read in place from the old repo's data/ folder; nothing is copied. Two checks:

  1. Live, same machine: the old repo's own pipeline (Keras ImageDataGenerator +
     flow_from_directory, then model.predict on its original .keras file, as its evaluate.py
     does) against this repo's preprocess + frozen model. Inputs must be bit-identical and every
     image must get the same class, with probabilities within TOLERANCE.
  2. Saved: this repo's predictions against the old repo's saved predictions.csv, same criteria.
Probabilities are not expected to be bit-equal: the old ones came from Metal GPU graph mode,
this repo runs eagerly (see load_frozen in src/base_model.py for why the results agree).
The old repo is only read.
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import preprocess  # noqa: E402
from src.base_model import BaseModel  # noqa: E402

N, SEED, TOLERANCE = 50, 42, 1e-5


def old_repo_inputs(old, files, size):
    """Model inputs made by the old repo's test generator, in `files` order.

    flow_from_directory needs class folders, so the 50 files are symlinked (not copied) into a
    temporary tree named 000_<file> to keep their order.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    tmp = Path(tempfile.mkdtemp(prefix="pbl3_match_"))
    try:
        (tmp / "all").mkdir()
        for i, f in enumerate(files):
            (tmp / "all" / f"{i:03d}_{Path(f).name}").symlink_to(old / f)
        gen = ImageDataGenerator(rescale=0.00392156862745098).flow_from_directory(
            tmp, target_size=size, color_mode="grayscale", class_mode=None, batch_size=len(files),
            shuffle=False)
        return next(gen)
    finally:
        shutil.rmtree(tmp)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-repo", required=True, type=Path)
    args = p.parse_args()
    old = args.old_repo.resolve()

    base = BaseModel("rafdb_clean")
    with open(old / "results" / "rafdb" / "clean" / "predictions.csv") as f:
        saved = list(csv.DictReader(f))
    sample = [saved[i] for i in sorted(np.random.default_rng(SEED).choice(len(saved), N, replace=False))]
    files = [row["file"] for row in sample]

    # This repo: preprocess each file, run the frozen model.
    x_new = np.concatenate([preprocess.to_model_input(preprocess.load_gray_file(old / f), base.input_size,
                                                      base.prep) for f in files])
    p_new = base.native_probs(x_new)

    # 1. Old repo's pipeline on its original (optimizer-included) model file.
    import keras

    x_old = old_repo_inputs(old, files, base.input_size)
    p_old = keras.models.load_model(old / "models" / "rafdb_clean.keras").predict(x_old, batch_size=32, verbose=0)
    inputs_equal = np.array_equal(x_old, x_new)
    same_live = (p_old.argmax(1) == p_new.argmax(1))
    diff_live = float(np.abs(p_old - p_new).max())
    print(f"[1] live, old pipeline vs new: inputs bit-identical={inputs_equal}, "
          f"same predicted class {same_live.sum()}/{N}, max |prob diff|={diff_live:.3g}")

    # 2. Saved predictions.csv (columns in the model's native order).
    cols = [f"p_{c}" for c in base.native_classes]
    p_saved = np.array([[float(row[c]) for c in cols] for row in sample], dtype=np.float32)
    same_class = (p_new.argmax(1) == p_saved.argmax(1))
    max_diff = float(np.abs(p_new - p_saved).max())
    print(f"[2] vs saved predictions.csv: same predicted class {same_class.sum()}/{N}, "
          f"max |prob diff|={max_diff:.3g} (tolerance {TOLERANCE:g})")
    correct = (np.array(base.native_classes)[p_new.argmax(1)] == [row["true"] for row in sample]).sum()
    print(f"    accuracy on these {N}: {correct}/{N}")

    ok = inputs_equal and same_live.all() and diff_live <= TOLERANCE and same_class.all() and max_diff <= TOLERANCE
    print("MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
