"""Log confirmed / corrected emotion examples for later per-user training.

    data/corrections/<user_id>/s01.jsonl   one JSON record per example
    data/corrections/<user_id>/s01.npy     float32 (n_examples, 512) feature vectors, row i = record i

Only numbers are stored: probabilities, features and labels. Never a face image or a frame.

The JSONL also holds recommendation events (key R in the study app), each with a "type":
    "correction"      one per confirmed/corrected example; has feature_row (the only kind in .npy)
    "recommendation"  songs shown (ids, rank, score) and the probabilities they were based on
    "rating"          a 1-5 rating of one shown song, pointing at its recommendation_id
Records without a "type" (older logs) are corrections.
After every example the .npy is rewritten atomically and then the JSONL line is appended, so a
crash cannot leave a record without its feature row (a stray extra feature row is harmless; the
`feature_row` field of each record is the authority).
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "corrections"

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
PSEUDONYM = re.compile(r"^[A-Za-z]{0,3}\d{1,4}$")  # P01, U7, 012 ...

NOTICE = """\
What is recorded: for each example you confirm or correct, the model's emotion probabilities,
a 512-number feature vector, your label, and a timestamp, averaged over the last {window} frames.
If you press R: the ids of the songs shown and any 1-5 ratings you give them.
No images or video are stored: camera frames stay in memory and are discarded.
Files go to: {folder}
  {jsonl}
  {npy}"""


def check_user_id(user_id):
    """Return a list of warnings; raise ValueError if the id cannot be used as a folder name."""
    if not SAFE_ID.match(user_id):
        raise ValueError(f"user id {user_id!r} must be 1-32 letters, digits, '_' or '-' (no spaces, '@' or '/')")
    if not PSEUDONYM.match(user_id):
        return [f"user id {user_id!r} does not look like a pseudonym such as P01. "
                "Use a code, not a real name."]
    return []


def next_session(log_dir, user_id):
    """Lowest unused session number for this user (1 if they have none yet)."""
    folder = Path(log_dir) / user_id
    used = [int(m.group(1)) for f in folder.glob("s*.jsonl") if (m := re.fullmatch(r"s(\d+)\.jsonl", f.name))] \
        if folder.exists() else []
    return max(used, default=0) + 1


class CorrectionLog:
    def __init__(self, user_id, session, model_name, model_sha256, class_order, feature_layer, feature_dim,
                 log_dir=DEFAULT_DIR):
        check_user_id(user_id)
        self.user_id, self.session = user_id, int(session)
        self.model_name, self.model_sha256 = model_name, model_sha256
        self.class_order, self.feature_layer, self.feature_dim = list(class_order), feature_layer, feature_dim
        folder = Path(log_dir) / user_id
        stem = f"s{self.session:02d}"
        self.jsonl_path, self.npy_path = folder / f"{stem}.jsonl", folder / f"{stem}.npy"
        if self.jsonl_path.exists():
            raise FileExistsError(f"{self.jsonl_path} already exists; pick another --session or omit it")
        folder.mkdir(parents=True, exist_ok=True)
        self._features = np.zeros((0, feature_dim), dtype=np.float32)
        self.counts = {"confirm": 0, "correct": 0}
        self.n_recommendations = 0

    def notice(self, window):
        return NOTICE.format(window=window, folder=self.jsonl_path.parent, jsonl=self.jsonl_path.name,
                             npy=self.npy_path.name)

    def add(self, probs, features, user_label, frames_averaged):
        """Log one example. `probs` are in class_order. Returns the record written."""
        probs, features = np.asarray(probs, dtype=np.float64), np.asarray(features, dtype=np.float32)
        assert probs.shape == (len(self.class_order),) and features.shape == (self.feature_dim,)
        if user_label not in self.class_order:
            raise ValueError(f"unknown label {user_label!r}; expected one of {self.class_order}")
        predicted = self.class_order[int(np.argmax(probs))]
        source = "confirm" if user_label == predicted else "correct"
        record = {
            "type": "correction",
            "user_id": self.user_id,
            "session_id": self.session,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "model": self.model_name,
            "model_sha256": self.model_sha256,
            "class_order": self.class_order,
            "feature_layer": self.feature_layer,
            "probabilities": [float(p) for p in probs],
            "predicted_label": predicted,
            "user_label": user_label,
            "source": source,
            "frames_averaged": int(frames_averaged),
            "feature_row": len(self._features),
        }
        self._features = np.vstack([self._features, features[None]])
        tmp = self.npy_path.with_name(self.npy_path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, self._features)
        os.replace(tmp, self.npy_path)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.counts[source] += 1
        return record


    def _append(self, record):
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _base(self, kind):
        return {"type": kind, "user_id": self.user_id, "session_id": self.session,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}

    def add_recommendation(self, recommender, probabilities, prob_source, language, recommendations):
        """Log the songs shown for one press of R. `recommendations` are src.recommend.Recommendation.
        Returns the recommendation_id (0, 1, 2 ... in this session) that ratings refer to."""
        rec_id = self.n_recommendations
        self.n_recommendations += 1
        self._append({
            **self._base("recommendation"), "recommendation_id": rec_id, "model": self.model_name,
            "recommender": recommender, "prob_source": prob_source, "language": language,
            "class_order": self.class_order, "probabilities": [float(p) for p in probabilities],
            "songs": [{"rank": i + 1, "song_id": r.song.song_id,
                       "score": None if r.score is None else float(r.score)}
                      for i, r in enumerate(recommendations)],
        })
        return rec_id

    def add_rating(self, recommendation_id, song_id, rating):
        """Log a 1-5 rating for a song shown in recommendation `recommendation_id`."""
        if not 1 <= int(rating) <= 5:
            raise ValueError(f"rating must be 1-5, got {rating!r}")
        record = {**self._base("rating"), "recommendation_id": int(recommendation_id), "song_id": song_id,
                  "rating": int(rating)}
        self._append(record)
        return record


def read_records(jsonl_path):
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_session(jsonl_path):
    """(correction records, features) of one session; features[r['feature_row']] belongs to record r.
    Recommendation and rating records are skipped (see load_events)."""
    jsonl_path = Path(jsonl_path)
    records = [r for r in read_records(jsonl_path) if r.get("type", "correction") == "correction"]
    npy = jsonl_path.with_suffix(".npy")
    features = np.load(npy) if npy.exists() else np.zeros((0, 0), dtype=np.float32)
    return records, features


def load_events(jsonl_path):
    """(recommendation records, rating records) of one session."""
    records = read_records(jsonl_path)
    return ([r for r in records if r.get("type") == "recommendation"],
            [r for r in records if r.get("type") == "rating"])
