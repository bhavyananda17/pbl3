# Per-user learning from corrections in a facial-expression music recommender

PBL research project. A webcam recognises the listener's facial expression and a music
recommender uses it; when the prediction is wrong the user corrects it, and a small per-user
model learns from those corrections. The general emotion model underneath stays frozen.

## Base model

The base model is the 7-class facial emotion CNN of **Raj & Demirkol, "Facial emotion
recognition using convolutional neural networks", *Scientific Reports* (2025)**. It was
reproduced and trained in my earlier repo,
[bhavyananda17/facial-emotion-recognition-cnn](https://github.com/bhavyananda17/facial-emotion-recognition-cnn),
at commit [`5d493ec`](https://github.com/bhavyananda17/facial-emotion-recognition-cnn/tree/5d493eced732c6faec635a9954e3e6e07dfd102e).
It is **not retrained here**: every layer is set non-trainable and it is loaded without an optimizer.

| Model | Input | Test accuracy | Trained at | Use |
|---|---|---|---|---|
| `rafdb_clean` | 100×100 grayscale | 83.2% (RAF-DB) | `b76b344` | default |
| `fer2013_clean` | 48×48 grayscale | 64.8% (FER2013) | `b76b344` | `BaseModel("fer2013_clean")` |

"clean" = early stopping on a held-out 10% of the training set, never on the test set.
The files in `models/base/` are the old repo's models re-saved without Adam optimizer state
(120 MB → 40 MB, 35 MB → 12 MB), with bit-identical weights; `scripts/export_base_models.py`
did this and checks it. Sizes, sha256 sums and provenance are in `configs/base.yaml`.

Preprocessing matches training exactly: PIL grayscale, `x * 1/255`, input size as above.
Live frames: OpenCV Haar cascade, largest face, resized with `INTER_AREA`.
Probabilities always come out in the order
`Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`, whichever model is loaded.

**Note on the first Dense layer.** The models were trained on tensorflow-metal, whose fused
MatMul+BiasAdd+ReLU kernel drops the ReLU in graph mode. The weights were therefore trained
as a network whose first Dense layer is linear, although the saved config says `relu`.
`src/base_model.py` sets it to linear, so results are the same on CPU, GPU, eager and graph mode.
Loading the file as saved and running it on a CPU drops RAF-DB accuracy from 83.2% to 67.8%.

## Setup

```bash
python3.11 -m venv .venv            # Python 3.11.15, as in the base-model repo
source .venv/bin/activate
pip install -r requirements.txt     # TensorFlow 2.18.1, Keras 3.8.0
pytest                              # freezing, class order, preprocessing
```

```python
from src.base_model import BaseModel
base = BaseModel()                          # rafdb_clean
probs, box = base.predict_frame(frame_bgr)  # (None, None) if no face
```

Check that predictions match the old repo (needs its local checkout with the RAF-DB data):

```bash
python scripts/check_match.py --old-repo ../facial-emotion-recognition-cnn
```

## Layout

```
src/base_model.py     load the frozen model, return 7 probabilities
src/preprocess.py     face detection and preprocessing (matches training)
configs/base.yaml     model paths, sha256, class order, preprocessing, provenance
models/base/          frozen base models (committed)
models/users/         per-user models (gitignored)
data/                 gitignored; data/corrections/ for user corrections
scripts/              model export and the match check against the old repo
tests/
```

No face images, video, datasets or per-user data are ever committed (see `.gitignore`).

## Planned work

- **Correction interface**: show the predicted emotion, let the user correct it, store the
  correction in `data/corrections/`.
- **Per-user layer**: a small model per user on top of the frozen base model's 512-d feature
  vector, trained from that user's corrections, saved in `models/users/`.
- **Recommendation step**: map the (personalised) emotion to music recommendations.
