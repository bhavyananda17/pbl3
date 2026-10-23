"""Recommender checks with a small synthetic song list.  Run: pytest"""

import csv

import numpy as np
import pytest

from src.base_model import load_config as load_base
from src.recommend import (LabelPlaylistRecommender, Song, VARecommender, load_config, load_songs,
                           make_recommender)

CLASSES = load_base()["classes"]
CFG = load_config()
COORDS = CFG["emotion_va"]


def onehot(label, conf=0.94):
    p = np.full(7, (1 - conf) / 6)
    p[CLASSES.index(label)] = conf
    return p


def songs():
    out = []
    for emo in ("Happy", "Sad"):
        for i, lang in enumerate(["en", "hi", "en", "hi"]):
            v, a = (0.9, 0.7) if emo == "Happy" else (0.1, 0.2)
            out.append(Song(f"{emo[0]}{i}", f"t{emo}{i}", "a", lang, emo, "u", v - 0.05 * i, a))
    return out


def ids(recs):
    return [r.song.song_id for r in recs]


def test_config_coordinates_cover_all_classes_in_range():
    assert set(COORDS) == set(CLASSES)
    assert all(len(v) == 2 and all(-1 <= x <= 1 for x in v) for v in COORDS.values())


# -- label recommender -----------------------------------------------------------------------
def test_label_picks_top_emotion():
    rec = LabelPlaylistRecommender(songs(), CLASSES)
    out = rec.recommend(onehot("Sad"), n=3)
    assert len(out) == 3 and {r.song.emotion for r in out} == {"Sad"} and all(r.score is None for r in out)


def test_label_language_filter():
    rec = LabelPlaylistRecommender(songs(), CLASSES)
    assert {r.song.language for r in rec.recommend(onehot("Happy"), n=5, language="hi")} == {"hi"}
    assert len(rec.recommend(onehot("Happy"), n=5, language="HI")) == 2  # case-insensitive, only 2 exist
    assert rec.recommend(onehot("Happy"), n=5, language="fr") == []


def test_label_excludes_recent_songs():
    rec = LabelPlaylistRecommender(songs(), CLASSES)
    out = rec.recommend(onehot("Happy"), n=5, exclude={"H0", "H1"})
    assert not {"H0", "H1"} & set(ids(out)) and len(out) == 2
    assert rec.recommend(onehot("Happy"), n=5, exclude={"H0", "H1", "H2", "H3"}) == []


def test_label_returns_fewer_than_n_not_other_emotions():
    out = LabelPlaylistRecommender(songs(), CLASSES).recommend(onehot("Happy"), n=10)
    assert len(out) == 4 and {r.song.emotion for r in out} == {"Happy"}


def test_label_is_reproducible_with_seed_and_no_duplicates():
    a = LabelPlaylistRecommender(songs(), CLASSES, seed=3).recommend(onehot("Happy"), n=3)
    b = LabelPlaylistRecommender(songs(), CLASSES, seed=3).recommend(onehot("Happy"), n=3)
    assert ids(a) == ids(b) and len(set(ids(a))) == 3


# -- valence-arousal recommender -------------------------------------------------------------
def make_va():
    return VARecommender(songs(), CLASSES, COORDS)


def test_va_target_is_probability_weighted_average():
    rec = make_va()
    p = np.zeros(7)
    p[CLASSES.index("Happy")] = p[CLASSES.index("Sad")] = 0.5
    expected = 0.5 * np.array(COORDS["Happy"]) + 0.5 * np.array(COORDS["Sad"])
    assert np.allclose(rec.target(p), expected)
    assert np.allclose(rec.target(onehot("Sad", 1.0)), COORDS["Sad"])


def test_va_returns_nearest_songs_in_order():
    rec = make_va()
    out = rec.recommend(onehot("Happy"), n=4)
    d = [r.score for r in out]
    assert d == sorted(d) and all(x is not None for x in d)
    assert all(r.song.emotion == "Happy" for r in out[:2])   # near the Happy point, not the Sad one
    assert rec.recommend(onehot("Sad"), n=1)[0].song.emotion == "Sad"


def test_va_rescales_song_values_with_2x_minus_1():
    s = [Song("a", "t", "a", "en", "Happy", "u", 1.0, 0.5), Song("b", "t", "a", "en", "Sad", "u", 0.0, 0.0)]
    rec = VARecommender(s, CLASSES, {**COORDS, "Happy": [1.0, 0.0], "Sad": [-1.0, -1.0]})
    top = rec.recommend(onehot("Happy", 1.0), n=2)
    assert top[0].song.song_id == "a" and top[0].score == pytest.approx(0.0)  # (1, 0) after rescaling
    assert top[1].score == pytest.approx(np.hypot(2, 1))


def test_va_language_and_exclude():
    rec = make_va()
    assert {r.song.language for r in rec.recommend(onehot("Happy"), n=8, language="hi")} == {"hi"}
    out = rec.recommend(onehot("Happy"), n=8, exclude={"H0", "H1"})
    assert not {"H0", "H1"} & set(ids(out)) and len(out) == 6


def test_va_needs_valence_and_energy():
    bad = [Song("a", "t", "a", "en", "Happy", "u")]
    with pytest.raises(ValueError, match="valence and energy"):
        VARecommender(bad, CLASSES, COORDS)


def test_bad_probabilities_rejected():
    for bad in (np.ones(6) / 6, -np.ones(7)):
        with pytest.raises(ValueError):
            LabelPlaylistRecommender(songs(), CLASSES).recommend(bad)


# -- CSV loading -----------------------------------------------------------------------------
def write_csv(path, rows, header=("song_id", "title", "artist", "language", "emotion", "url", "valence", "energy")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_example_csv_loads_and_works_with_both_recommenders():
    for kind in ("label", "va"):
        rec = make_recommender(kind, CFG, CLASSES)
        out = rec.recommend(onehot("Happy"), n=CFG["n"])
        assert len(out) == 3 if kind == "label" else len(out) == CFG["n"]
    all_songs = load_songs(CFG["songs_csv"], CLASSES, need_va=True)
    assert len(all_songs) == 21 and all("EXAMPLE" in s.title and "EXAMPLE" in s.artist for s in all_songs)


def test_csv_validation(tmp_path):
    ok = ["s1", "t", "a", "en", "Happy", "u", "0.5", "0.5"]
    assert len(load_songs(write_csv(tmp_path / "ok.csv", [ok]), CLASSES, need_va=True)) == 1
    with pytest.raises(ValueError, match="emotion"):
        load_songs(write_csv(tmp_path / "a.csv", [ok[:4] + ["Joy"] + ok[5:]]), CLASSES)
    with pytest.raises(ValueError, match="duplicated"):
        load_songs(write_csv(tmp_path / "b.csv", [ok, ok]), CLASSES)
    with pytest.raises(ValueError, match="outside"):
        load_songs(write_csv(tmp_path / "c.csv", [ok[:6] + ["1.5", "0.5"]]), CLASSES)
    with pytest.raises(ValueError, match="missing column"):
        load_songs(write_csv(tmp_path / "d.csv", [ok[:6]], header=("song_id", "title", "artist", "language",
                                                                   "emotion", "url")), CLASSES, need_va=True)
    # valence/energy are optional for the label recommender
    assert load_songs(write_csv(tmp_path / "e.csv", [ok[:6]], header=("song_id", "title", "artist", "language",
                                                                      "emotion", "url")), CLASSES)[0].valence is None
