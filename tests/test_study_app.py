"""Frame averaging and key handling with synthetic inputs (no camera).  Run: pytest"""

import numpy as np
import pytest

from src.base_model import load_config
from src.correction_log import CorrectionLog, load_session
from src.study_app import FrameWindow, handle_key, key_map

CLASSES = load_config()["classes"]
W = load_config()["study"]["correction_window"]


def onehot(label, conf=0.9):
    p = np.full(7, (1 - conf) / 6)
    p[CLASSES.index(label)] = conf
    return p


def filled(n, label="Happy", feat_value=1.0):
    win = FrameWindow(W)
    for _ in range(n):
        win.push(onehot(label), np.full(512, feat_value))
    return win


@pytest.fixture
def log(tmp_path):
    return CorrectionLog("P01", 1, "rafdb_clean", "sha", CLASSES, "batch_normalization_8", 512, tmp_path)


def test_window_size_comes_from_config():
    assert W == 10


def test_mean_over_last_w_frames_only():
    win = FrameWindow(W)
    for i in range(25):  # 15 old frames of 0 then 10 of 1: only the last 10 count
        win.push(np.full(7, 1 / 7), np.full(512, 0.0 if i < 15 else 1.0))
    probs, feats, n = win.mean()
    assert n == W and win.ready and np.allclose(feats, 1.0)


def test_mean_value():
    win = FrameWindow(2)
    win.push(np.array([1, 0, 0, 0, 0, 0, 0]), np.zeros(512))
    win.push(np.array([0, 1, 0, 0, 0, 0, 0]), np.full(512, 2.0))
    probs, feats, n = win.mean()
    assert np.allclose(probs, [0.5, 0.5, 0, 0, 0, 0, 0]) and np.allclose(feats, 1.0) and n == 2


def test_empty_and_clear():
    win = filled(W)
    win.clear()
    assert len(win) == 0 and not win.ready and win.mean() is None


def test_too_few_frames_logs_nothing_and_warns(log):
    win = filled(W - 1)
    msg, warn, quit_ = handle_key(" ", win, CLASSES, log)
    assert warn and "Not logged" in msg and not quit_
    assert not log.jsonl_path.exists() and log.counts == {"confirm": 0, "correct": 0}
    assert len(win) == W - 1  # frames are kept, the user can keep holding still


def test_space_confirms_prediction(log):
    win = filled(W, "Sad", 3.0)
    msg, warn, _ = handle_key(" ", win, CLASSES, log)
    records, feats = load_session(log.jsonl_path)
    assert not warn and records[0]["user_label"] == records[0]["predicted_label"] == "Sad"
    assert records[0]["source"] == "confirm" and records[0]["frames_averaged"] == W
    assert np.allclose(feats[0], 3.0)
    assert len(win) == 0  # next example needs fresh frames


def test_number_key_corrects_using_config_order(log):
    for key, label in key_map(CLASSES).items():
        assert CLASSES[int(key) - 1] == label
    win = filled(W, "Happy")
    handle_key("1", win, CLASSES, log)  # key 1 = first class in the config
    r = load_session(log.jsonl_path)[0][0]
    assert r["user_label"] == CLASSES[0] and r["predicted_label"] == "Happy"
    assert r["source"] == ("confirm" if CLASSES[0] == "Happy" else "correct")


def test_number_key_matching_prediction_is_a_confirm(log):
    win = filled(W, "Fear")
    handle_key(str(CLASSES.index("Fear") + 1), win, CLASSES, log)
    assert load_session(log.jsonl_path)[0][0]["source"] == "confirm"


def test_skip_quit_and_unknown_keys(log):
    win = filled(W)
    assert handle_key("s", win, CLASSES, log)[2] is False and len(win) == W
    assert handle_key("x", win, CLASSES, log) == (None, False, False)
    assert handle_key("8", win, CLASSES, log) == (None, False, False)
    assert handle_key("q", win, CLASSES, log)[2] is True
    assert not log.jsonl_path.exists()
