"""Recommend songs from a local CSV for the (personalised) emotion probabilities.

    rec = make_recommender("va", cfg, classes)
    rec.recommend(probabilities, n=5, language="en", exclude=recent_ids)   # -> [Recommendation, ...]

The recommenders are interchangeable: same constructor inputs, same recommend() signature.
No network is used (Spotify's recommendation and audio-feature endpoints are closed to new apps);
songs come only from a CSV with these columns:

    song_id, title, artist, language, emotion, url      required
    valence, energy                                      optional, each in [0, 1]; required by "va"

`emotion` must be one of the classes in configs/base.yaml. Probabilities are in that class order.
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "recommend.yaml"
REQUIRED = ["song_id", "title", "artist", "language", "emotion", "url"]


@dataclass(frozen=True)
class Song:
    song_id: str
    title: str
    artist: str
    language: str
    emotion: str
    url: str
    valence: float = None
    energy: float = None


class Recommendation(NamedTuple):
    song: Song
    score: float  # distance to the target point for "va" (smaller = closer); None for "label"


def load_config(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def load_songs(path, classes, need_va=False):
    """Read and validate a songs CSV. Raises ValueError with the row number on bad data."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED + (["valence", "energy"] if need_va else []) if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: missing column(s) {missing}; required: {REQUIRED}")
        songs, seen = [], set()
        for line, row in enumerate(reader, start=2):
            where = f"{path.name} line {line}"
            if row["emotion"] not in classes:
                raise ValueError(f"{where}: emotion {row['emotion']!r} is not one of {list(classes)}")
            if row["song_id"] in seen or not row["song_id"]:
                raise ValueError(f"{where}: song_id {row['song_id']!r} is empty or duplicated")
            seen.add(row["song_id"])
            va = {}
            for col in ("valence", "energy"):
                raw = (row.get(col) or "").strip()
                if raw == "":
                    if need_va:
                        raise ValueError(f"{where}: {col} is empty")
                    va[col] = None
                    continue
                va[col] = float(raw)
                if not 0.0 <= va[col] <= 1.0:
                    raise ValueError(f"{where}: {col} {va[col]} is outside [0, 1]")
            songs.append(Song(**{k: row[k].strip() for k in REQUIRED}, **va))
    return songs


def _check_probabilities(probabilities, classes):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(classes),) or (p < 0).any():
        raise ValueError(f"probabilities must be {len(classes)} non-negative numbers in the config class order")
    return p


class Recommender:
    kind = "base"

    def __init__(self, songs, classes):
        self.songs, self.classes = list(songs), list(classes)

    def _candidates(self, language, exclude):
        exclude = set(exclude or ())
        lang = language.strip().lower() if language else None
        return [s for s in self.songs if s.song_id not in exclude and (lang is None or s.language.lower() == lang)]

    def recommend(self, probabilities, n=5, language=None, exclude=None):
        raise NotImplementedError


class LabelPlaylistRecommender(Recommender):
    """Songs labelled with the most probable emotion; a seeded random pick among them."""

    kind = "label"

    def __init__(self, songs, classes, seed=0):
        super().__init__(songs, classes)
        self._rng = random.Random(seed)

    def recommend(self, probabilities, n=5, language=None, exclude=None):
        top = self.classes[int(np.argmax(_check_probabilities(probabilities, self.classes)))]
        matches = [s for s in self._candidates(language, exclude) if s.emotion == top]
        picked = self._rng.sample(matches, min(n, len(matches)))
        return [Recommendation(s, None) for s in picked]


class VARecommender(Recommender):
    """Nearest songs to the probability-weighted valence-arousal point of the emotion probabilities."""

    kind = "va"

    def __init__(self, songs, classes, emotion_va):
        super().__init__(songs, classes)
        missing = [c for c in classes if c not in emotion_va]
        if missing:
            raise ValueError(f"emotion_va has no coordinates for {missing}")
        self.coords = np.array([emotion_va[c] for c in classes], dtype=np.float64)  # (7, 2), in [-1, 1]
        if any(s.valence is None or s.energy is None for s in self.songs):
            raise ValueError("the va recommender needs valence and energy for every song")

    def target(self, probabilities):
        """The valence-arousal point of these probabilities (weights are normalised to sum to 1)."""
        p = _check_probabilities(probabilities, self.classes)
        return (p / p.sum()) @ self.coords

    def recommend(self, probabilities, n=5, language=None, exclude=None):
        t = self.target(probabilities)
        scored = [(float(np.hypot(2 * s.valence - 1 - t[0], 2 * s.energy - 1 - t[1])), s)
                  for s in self._candidates(language, exclude)]
        scored.sort(key=lambda ds: (ds[0], ds[1].song_id))
        return [Recommendation(s, d) for d, s in scored[:n]]


def make_recommender(kind, cfg, classes, songs_csv=None):
    """Build a recommender from configs/recommend.yaml (`cfg`); `songs_csv` overrides the config path."""
    songs = load_songs(songs_csv or cfg["songs_csv"], classes, need_va=(kind == "va"))
    if kind == "label":
        return LabelPlaylistRecommender(songs, classes, seed=cfg["seed"])
    if kind == "va":
        return VARecommender(songs, classes, cfg["emotion_va"])
    raise ValueError(f"unknown recommender {kind!r}; choose label or va")
