"""Session split, learning-curve prefixes and metrics on small synthetic logs.  Run: pytest"""

import numpy as np
import pytest

from src.base_model import load_config
from src.correction_log import CorrectionLog
from src.evaluation import (average_over_users, evaluate, load_logs, main_splits, resolve_ks, score)

CFG = load_config()
CLASSES = CFG["classes"]
DIM = 512


def write_session(tmp_path, user, session, labels, seed, model="rafdb_clean", sha="sha-a"):
    """Synthetic session: each class has its own direction; the generic model is right for ~half."""
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(99).normal(size=(7, DIM))
    log = CorrectionLog(user, session, model, sha, CLASSES, "batch_normalization_8", DIM, tmp_path)
    for i, label in enumerate(labels):
        feat = centers[CLASSES.index(label)] + 0.3 * rng.normal(size=DIM)
        pred = label if i % 2 == 0 else CLASSES[(CLASSES.index(label) + 1) % 7]
        p = np.full(7, 0.04)
        p[CLASSES.index(pred)] = 0.76
        log.add(p, feat, label, 10)


def cycle(n):
    return [CLASSES[i % 7] for i in range(n)]


def test_split_never_tests_on_a_trained_session(tmp_path):
    for s in (1, 2, 3):
        write_session(tmp_path, "P01", s, cycle(7), seed=s)
    sessions = next(iter(load_logs(tmp_path, CLASSES).values()))
    splits = main_splits(sessions)
    assert [t["session"] for t, _ in splits] == [2, 3]
    for test, train in splits:
        assert test["session"] not in [s["session"] for s in train]
        assert all(s["session"] < test["session"] for s in train)


def test_resolve_ks():
    assert resolve_ks([0, 1, 2, 4, 8, 16, "all"], 10) == [0, 1, 2, 4, 8, 10]
    assert resolve_ks([0, 1, 2, 4, 8, 16, "all"], 0) == [0]
    assert resolve_ks([0, 1, 2, 4, 8, 16, "all"], 16) == [0, 1, 2, 4, 8, 16]


def test_main_protocol_and_learning_curve(tmp_path):
    write_session(tmp_path, "P01", 1, cycle(14), seed=1)
    write_session(tmp_path, "P01", 2, cycle(14), seed=2)
    rows, sessions, warnings = evaluate(load_logs(tmp_path, CLASSES), CLASSES, CFG)
    main = [r for r in rows if r["protocol"] == "main"]
    assert {r["k"] for r in main} == {0, 1, 2, 4, 8, 14}
    assert {r["method"] for r in main} == {"generic", "logistic", "prototype", "logistic+generic",
                                           "prototype+generic"}
    acc = {(r["k"], r["method"]): r["accuracy"] for r in main}
    for m in acc:  # k = 0: every method is the generic model
        if m[0] == 0:
            assert acc[m] == acc[(0, "generic")] == pytest.approx(0.5)
    assert acc[(14, "prototype")] == 1.0            # separable synthetic classes are learned
    assert acc[(14, "prototype+generic")] > acc[(0, "generic")]
    assert [r["correction_rate"] for r in sessions] == [0.5, 0.5]
    assert any("anecdotal" in w for w in warnings)  # tiny data is flagged


def test_single_session_user_gets_online_protocol_and_warning(tmp_path):
    write_session(tmp_path, "P01", 1, cycle(21), seed=1)
    rows, _, warnings = evaluate(load_logs(tmp_path, CLASSES), CLASSES, CFG)
    assert {r["protocol"] for r in rows} == {"online"}
    assert any("only 1 session" in w for w in warnings)
    online = {(r["k"], r["method"]): r for r in rows}
    assert online[(0, "generic")]["n_test"] == 1 and online[(16, "generic")]["n_test"] == 5
    assert sum(r["n_test"] for r in rows if r["method"] == "generic") == 21


def test_online_uses_only_earlier_examples(tmp_path):
    """The first example has nothing before it, so every method must equal the generic model there."""
    write_session(tmp_path, "P01", 1, cycle(8), seed=1)
    rows, _, _ = evaluate(load_logs(tmp_path, CLASSES), CLASSES, CFG)
    first = [r for r in rows if r["k"] == 0 and r["n_test"] == 1]
    assert len({r["accuracy"] for r in first}) == 1


def test_users_and_models_are_not_mixed(tmp_path):
    write_session(tmp_path, "P01", 1, cycle(7), seed=1, model="rafdb_clean", sha="sha-a")
    write_session(tmp_path, "P01", 2, cycle(7), seed=2, model="fer2013_clean", sha="sha-b")
    write_session(tmp_path, "P02", 1, cycle(7), seed=3)
    groups = load_logs(tmp_path, CLASSES)
    assert len(groups) == 3 and all(len(v) == 1 for v in groups.values())


def test_wrong_class_order_rejected(tmp_path):
    write_session(tmp_path, "P01", 1, cycle(3), seed=1)
    with pytest.raises(ValueError, match="class order"):
        load_logs(tmp_path, CLASSES[::-1])


def test_score_and_missing_class_recall():
    s = score([0, 0, 1, 3], [0, 1, 1, 3], CLASSES)
    assert s["accuracy"] == 0.75 and s["recall"]["Angry"] == 0.5 and s["recall"]["Disgust"] == 1.0
    assert np.isnan(s["recall"]["Fear"])  # no true Fear examples


def test_average_over_users_is_unweighted(tmp_path):
    write_session(tmp_path, "P01", 1, cycle(7), seed=1)
    write_session(tmp_path, "P01", 2, cycle(7), seed=2)
    write_session(tmp_path, "P02", 1, cycle(7), seed=3)
    write_session(tmp_path, "P02", 2, cycle(7), seed=4)
    rows, _, _ = evaluate(load_logs(tmp_path, CLASSES), CLASSES, CFG)
    avg = average_over_users(rows, CLASSES)
    a = [r for r in avg if r["k"] == 0 and r["method"] == "generic"][0]
    assert a["n_users"] == 2
