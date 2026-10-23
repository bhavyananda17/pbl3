"""Log confirmed / corrected emotion examples for later per-user training.

    data/corrections/<user_id>/s01.jsonl   one JSON record per example
    data/corrections/<user_id>/s01.npy     float32 (n_examples, 512) feature vectors, row i = record i

Only numbers are stored: probabilities, features and labels. Never a face image or a frame.
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


def load_session(jsonl_path):
    """(records, features) of one session; features[r['feature_row']] belongs to record r."""
    jsonl_path = Path(jsonl_path)
    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    features = np.load(jsonl_path.with_suffix(".npy"))
    return records, features
