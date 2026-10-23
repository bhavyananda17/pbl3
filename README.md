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

## Running a study session

```bash
python -m src.study_app --user-id P01                    # RAF-DB model, default camera
python -m src.study_app --user-id P01 --model fer2013_clean --camera 1
python -m src.study_app --list-cameras                   # find the Mac webcam (Continuity Camera can be 0)
```

`--user-id` is a pseudonym such as `P01` (a warning appears if it looks like a real name; e-mail
addresses and spaces are rejected). `--session` defaults to the next unused number for that user.
Run it from the macOS Terminal app and allow camera access. The window shows the detected emotion,
per-class bars, the key map, face-frame count and confirm/correct counts.

| Key | Action |
|---|---|
| `1`-`7` | the correct emotion, in the order of `classes` in `configs/base.yaml` (1 Angry, 2 Disgust, 3 Fear, 4 Happy, 5 Neutral, 6 Sad, 7 Surprise) |
| Space | the prediction is correct |
| `S` | skip |
| `Q` | quit |

A keypress logs the probabilities and 512-d features averaged over the last `study.correction_window`
(10) frames that had a face. With fewer than 10, nothing is logged and a warning is shown. Frames are
cleared when the face is lost and after each logged example, so every example uses fresh frames.

Output, per user and session (`data/corrections/` is gitignored):

```
data/corrections/P01/s01.jsonl   one record per example: user_id, session_id, timestamp, model,
                                 model_sha256, class_order, feature_layer, probabilities,
                                 predicted_label, user_label, source (confirm|correct),
                                 frames_averaged, feature_row
data/corrections/P01/s01.npy     float32 (n, 512); row `feature_row` belongs to each record
```

Features are the output of `batch_normalization_8`, the input to the final softmax layer.
No face images or frames are ever saved; a notice saying so is printed at start-up.

## Per-user layer and its evaluation

`src/personalize.py` has two adapters that learn from a user's corrections on the frozen model's
512-d features (the base model is never changed):

- `LogisticAdapter`: multinomial logistic regression, strong L2 (`C`), balanced class weights.
- `PrototypeAdapter`: L2-normalised mean feature per class, cosine similarity, softmax with a temperature.
- `blend(p_user, p_generic, n, k0)`: `p = w * p_user + (1 - w) * p_generic`, `w = n / (n + k0)`,
  where `n` is the user's number of labelled examples.

Classes the user has not labelled yet keep the generic model's probability (the adapter's output over
the classes it has seen is rescaled to the generic mass on them). With fewer than two seen classes the
logistic adapter returns the generic probabilities. Fitted adapters are saved with
`adapter.save(user_id, model, sha256)` under `models/users/<user_id>/` (gitignored) and refused on
load if the base model file or class order differs. Settings (`k0`, `C`, temperature, the learning
curve `ks`) are in `configs/base.yaml` under `personalize`; they are untuned defaults, so do not tune
them on the sessions you report.

```bash
python scripts/evaluate_user_logs.py                       # all logs in data/corrections/
python scripts/evaluate_user_logs.py --log-dir data/corrections --out results/personalization
```

**Protocol.** For each user (and base model; features of different models are never mixed), train on
the sessions before session s and test on s, so a session is never tested after being learned from.
The learning curve trains on only the first k examples of that training pool, k in 0, 1, 2, 4, 8, 16,
all (k = 0 is the generic model). Methods: generic, logistic, prototype, and each blended with generic.
Reported: accuracy, macro-F1 (over classes present in truth or predictions), per-class recall (NaN where a
class has no test examples; look at Sad, Fear, Disgust), and correction rate per session. Per-user numbers
pool that user's test examples; averages are unweighted over users. Output: `results.csv`, `results.json`,
`sessions.csv` and `accuracy_vs_k.png` in `results/personalization/`.

**Online protocol.** A user with a single session has nothing earlier to train on, so the script reports
a separate, clearly labelled "online" result: each example is predicted by adapters trained on that user's
earlier examples in the same session. It shows a learning curve, but it is not the main protocol.

The script warns when a user has too few examples or sessions, when classes have no examples, when a test
session is small, and when the average is over fewer than 3 users. Heed those: with a handful of examples
the numbers are anecdotes, not results.

## Recommending songs (key R)

`src/recommend.py` turns the (personalised) emotion probabilities into songs from a **local CSV**. No
web API is called (Spotify's recommendation and audio-feature endpoints are closed to new apps). Two
interchangeable recommenders, chosen by `recommender:` in `configs/recommend.yaml` or `--recommender`:

- `label`: random pick (seeded) among songs labelled with the most probable emotion.
- `va`: the probabilities become a valence-arousal point, the probability-weighted average of the
  per-emotion coordinates in `configs/recommend.yaml`; songs' valence and energy (0-1) are rescaled with
  `2x - 1` and the `n` nearest songs (Euclidean) are returned.

Both take `recommend(probabilities, n=5, language=None, exclude=recent_ids)`. `language` keeps only
songs in that language, `exclude` skips song ids (the study app excludes the last 20 songs shown).
If fewer than `n` songs qualify you get fewer, never songs of another emotion.

**Valence-arousal coordinates** (valence, arousal in [-1, 1]): Happy (0.8, 0.5), Surprise (0.2, 0.8),
Angry (-0.6, 0.8), Fear (-0.7, 0.6), Disgust (-0.6, 0.3), Sad (-0.8, -0.5), Neutral (0, 0). They are
approximate hand placements on Russell's (1980) circumplex and **must be justified in the paper** (cite
a source or validate them); the results of the `va` recommender depend on them.

**Song CSV schema** (`configs/example_songs.csv` is a fake example; every row is labelled EXAMPLE):

| column | |
|---|---|
| `song_id` | unique id (required) |
| `title`, `artist`, `url` | text (required) |
| `language` | e.g. `en`, `hi`; compared case-insensitively (required) |
| `emotion` | one of the classes in `configs/base.yaml` (required) |
| `valence`, `energy` | numbers in [0, 1]; optional, required by `va` |

Point `songs_csv` in `configs/recommend.yaml` (or `--songs path.csv`) at your dataset. Rows are
validated at start-up with the line number of any error.

In the study app, **R** shows `n` songs (`--language en` to filter) for the 10-frame average, like the
correction keys (it warns if fewer than 10 face frames). The list stays open: `1`-`5` rates the highlighted
song and moves on, `N` moves on without rating, `R` or `Esc` closes it (the emotion keys do nothing meanwhile).
Probabilities come from the user's saved adapter if there is one, otherwise the generic model; the list
shows which. To create adapters from a user's logs (all their sessions):

```bash
python scripts/fit_user_adapter.py --user-id P01          # saves to models/users/P01/ (gitignored)
python -m src.study_app --user-id P01 --recommender va --language en
```

Which adapter output R uses is `probabilities:` in `configs/recommend.yaml` (default `logistic+generic`).
Each press of R adds a `"type": "recommendation"` record (song ids, ranks, scores, the probabilities used and
whether they came from an adapter) to the session's `.jsonl`; each rating adds a `"type": "rating"` record
pointing at its `recommendation_id`. They have no row in the `.npy`, and `evaluate_user_logs.py` ignores them.
Only song ids are stored, never images.

## Layout

```
src/base_model.py     load the frozen model, return 7 probabilities
src/preprocess.py     face detection and preprocessing (matches training)
src/correction_log.py JSONL + .npy log of confirmed / corrected examples
src/study_app.py      OpenCV study app (confirm or correct the detected emotion)
src/personalize.py    per-user adapters, blending, save/load
src/recommend.py      label and valence-arousal song recommenders (local CSV)
src/evaluation.py     evaluation protocol on correction logs
configs/base.yaml     model paths, sha256, class order, preprocessing, provenance
configs/recommend.yaml recommender settings, valence-arousal coordinates; example_songs.csv
models/base/          frozen base models (committed)
models/users/         per-user models (gitignored)
data/                 gitignored; data/corrections/ for user corrections
scripts/              model export, match check, evaluate_user_logs.py, fit_user_adapter.py
tests/
```

No face images, video, datasets or per-user data are ever committed (see `.gitignore`).

## Planned work

- ~~**Correction interface**~~ done: `src/study_app.py` logs corrections to `data/corrections/`.
- ~~**Per-user layer**~~ done: `src/personalize.py`, evaluated by `scripts/evaluate_user_logs.py`.
- ~~**Recommendation step**~~ done: `src/recommend.py`, key R in the study app. Needs a real song dataset.
