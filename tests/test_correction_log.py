"""Logger checks with synthetic data.  Run: pytest"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.correction_log import CorrectionLog, check_user_id, load_session, next_session

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def make_log(tmp_path, user="P01", session=1):
    return CorrectionLog(user, session, "rafdb_clean", "abc123", CLASSES, "batch_normalization_8", 512, tmp_path)


def probs_for(label):
    p = np.full(7, 0.02)
    p[CLASSES.index(label)] = 0.88
    return p


def test_round_trip_and_alignment(tmp_path):
    log = make_log(tmp_path)
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(3, 512)).astype(np.float32)
    plan = [("Happy", "Happy"), ("Sad", "Neutral"), ("Angry", "Fear")]  # (model says, user says)
    for f, (pred, user) in zip(feats, plan):
        log.add(probs_for(pred), f, user, 10)
    records, loaded = load_session(log.jsonl_path)
    assert len(records) == 3 and loaded.shape == (3, 512) and loaded.dtype == np.float32
    for i, (r, (pred, user)) in enumerate(zip(records, plan)):
        assert r["feature_row"] == i and np.array_equal(loaded[r["feature_row"]], feats[i])
        assert r["predicted_label"] == pred and r["user_label"] == user
        assert r["source"] == ("confirm" if pred == user else "correct")
        assert r["class_order"] == CLASSES and r["probabilities"] == pytest.approx(probs_for(pred).tolist())
        assert (r["user_id"], r["session_id"], r["model"], r["model_sha256"], r["frames_averaged"]) == \
               ("P01", 1, "rafdb_clean", "abc123", 10)
    assert log.counts == {"confirm": 1, "correct": 2}


def test_files_stay_aligned_after_every_write(tmp_path):
    log = make_log(tmp_path)
    for i in range(4):
        log.add(probs_for("Happy"), np.full(512, i, dtype=np.float32), "Happy", 10)
        records, loaded = load_session(log.jsonl_path)
        assert len(records) == len(loaded) == i + 1
        assert loaded[i, 0] == i


def test_session_auto_increment(tmp_path):
    assert next_session(tmp_path, "P01") == 1
    make_log(tmp_path, session=1).add(probs_for("Happy"), np.zeros(512), "Happy", 10)
    assert next_session(tmp_path, "P01") == 2
    make_log(tmp_path, session=2).add(probs_for("Happy"), np.zeros(512), "Happy", 10)
    assert next_session(tmp_path, "P01") == 3
    assert next_session(tmp_path, "P02") == 1  # other users are independent


def test_existing_session_is_never_overwritten(tmp_path):
    make_log(tmp_path).add(probs_for("Happy"), np.zeros(512), "Happy", 10)
    with pytest.raises(FileExistsError):
        make_log(tmp_path)


def test_unknown_label_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_log(tmp_path).add(probs_for("Happy"), np.zeros(512), "Joy", 10)


def test_only_jsonl_and_npy_written(tmp_path):
    make_log(tmp_path).add(probs_for("Happy"), np.zeros(512), "Happy", 10)
    names = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    assert names == ["s01.jsonl", "s01.npy"]


def test_user_id_checks():
    assert check_user_id("P01") == []
    assert check_user_id("alice") and check_user_id("Bhavya_Nanda")  # look like names -> warning
    for bad in ("a@b.com", "../x", "P 01", "", "x" * 40):
        with pytest.raises(ValueError):
            check_user_id(bad)


def test_corrections_folder_is_gitignored():
    root = Path(__file__).resolve().parent.parent
    for path in ("data/corrections/P01/s01.jsonl", "data/corrections/P01/s01.npy"):
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=root)
        assert r.returncode == 0, f"{path} is not gitignored"


# -- recommendation events ---------------------------------------------------------------------
def fake_recs(n=3):
    from src.recommend import Recommendation, Song

    return [Recommendation(Song(f"S{i}", "t", "a", "en", "Happy", "u"), 0.1 * i if i else None) for i in range(n)]


def test_recommendation_and_rating_records(tmp_path):
    from src.correction_log import load_events

    log = make_log(tmp_path)
    rid = log.add_recommendation("va", probs_for("Happy"), "adapter:logistic+generic", "en", fake_recs())
    rid2 = log.add_recommendation("va", probs_for("Sad"), "generic", None, fake_recs(1))
    log.add_rating(rid, "S1", 4)
    recs, ratings = load_events(log.jsonl_path)
    assert (rid, rid2) == (0, 1) and [r["recommendation_id"] for r in recs] == [0, 1]
    assert recs[0]["prob_source"] == "adapter:logistic+generic" and recs[0]["language"] == "en"
    assert [s["song_id"] for s in recs[0]["songs"]] == ["S0", "S1", "S2"] and recs[0]["songs"][0]["rank"] == 1
    assert ratings[0]["rating"] == 4 and ratings[0]["song_id"] == "S1" and ratings[0]["recommendation_id"] == 0
    assert all("feature_row" not in r for r in recs + ratings)


@pytest.mark.parametrize("bad", [0, 6, -1])
def test_rating_must_be_1_to_5(tmp_path, bad):
    with pytest.raises(ValueError):
        make_log(tmp_path).add_rating(0, "S1", bad)


def test_events_do_not_disturb_corrections_or_alignment(tmp_path):
    log = make_log(tmp_path)
    log.add_recommendation("label", probs_for("Happy"), "generic", None, fake_recs())
    log.add(probs_for("Happy"), np.full(512, 1.0, dtype=np.float32), "Happy", 10)
    log.add_recommendation("label", probs_for("Happy"), "generic", None, fake_recs())
    log.add(probs_for("Sad"), np.full(512, 2.0, dtype=np.float32), "Sad", 10)
    records, feats = load_session(log.jsonl_path)
    assert [r["type"] for r in records] == ["correction", "correction"] and feats.shape == (2, 512)
    assert [feats[r["feature_row"], 0] for r in records] == [1.0, 2.0]


def test_session_with_only_recommendations_loads_and_is_skipped_by_evaluation(tmp_path):
    from src.base_model import load_config
    from src.evaluation import load_logs

    log = make_log(tmp_path)
    log.add_recommendation("label", probs_for("Happy"), "generic", None, fake_recs())
    assert not log.npy_path.exists()
    assert load_session(log.jsonl_path)[0] == []
    assert load_logs(tmp_path, load_config()["classes"]) == {}


def test_old_records_without_type_still_count_as_corrections(tmp_path):
    log = make_log(tmp_path)
    log.add(probs_for("Happy"), np.zeros(512, dtype=np.float32), "Happy", 10)
    lines = [json.loads(l) for l in open(log.jsonl_path)]
    for r in lines:
        r.pop("type")
    log.jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    assert len(load_session(log.jsonl_path)[0]) == 1
