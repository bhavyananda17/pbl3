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


# -- key R: recommendations and ratings ------------------------------------------------------------
from src.personalize import PrototypeAdapter, personalised_probabilities  # noqa: E402
from src.recommend import LabelPlaylistRecommender, Song, VARecommender, load_config as load_rec_config  # noqa: E402
from src.correction_log import load_events  # noqa: E402
from src.study_app import RecContext, handle_rec_key, start_recommendation  # noqa: E402

RCFG = load_rec_config()


def song_list():
    return [Song(f"{e[0]}{i}", f"{e} {i}", "a", "en" if i % 2 else "hi", e, "u", 0.5, 0.5)
            for e in CLASSES for i in range(4)]


def make_ctx(adapters=None, choice="logistic+generic", n=3, language=None, recent=20, kind="label"):
    rec = (LabelPlaylistRecommender(song_list(), CLASSES, seed=0) if kind == "label"
           else VARecommender(song_list(), CLASSES, RCFG["emotion_va"]))
    return RecContext(rec, CLASSES, adapters or {}, choice, 8, n, language, recent)


def test_r_needs_a_full_window(log):
    state, msg, warn = start_recommendation(filled(W - 1), make_ctx(), log)
    assert state is None and warn and "face frames" in msg
    assert not log.jsonl_path.exists()


def test_r_shows_n_songs_for_top_emotion_and_logs_them(log):
    ctx = make_ctx(n=3)
    state, msg, warn = start_recommendation(filled(W, "Sad"), ctx, log)
    assert not warn and len(state.recs) == 3 and {r.song.emotion for r in state.recs} == {"Sad"}
    recs, _ = load_events(log.jsonl_path)
    assert recs[0]["prob_source"] == "generic" and recs[0]["recommender"] == "label"
    assert [s["song_id"] for s in recs[0]["songs"]] == [r.song.song_id for r in state.recs]
    assert list(ctx.recent) == [r.song.song_id for r in state.recs]


def test_second_r_excludes_songs_already_shown(log):
    ctx = make_ctx(n=3)
    first, _, _ = start_recommendation(filled(W, "Sad"), ctx, log)
    second, _, _ = start_recommendation(filled(W, "Sad"), ctx, log)
    shown = {r.song.song_id for r in first.recs}
    assert len(second.recs) == 1 and not shown & {r.song.song_id for r in second.recs}  # 4 Sad songs in total
    third, msg, warn = start_recommendation(filled(W, "Sad"), ctx, log)
    assert third is None and warn and "No songs" in msg


def test_language_flag_filters(log):
    state, _, _ = start_recommendation(filled(W, "Happy"), make_ctx(n=5, language="hi"), log)
    assert {r.song.language for r in state.recs} == {"hi"}
    assert load_events(log.jsonl_path)[0][0]["language"] == "hi"


def test_r_with_va_recommender(log):
    state, _, _ = start_recommendation(filled(W, "Happy"), make_ctx(kind="va"), log)
    assert len(state.recs) == 3 and all(r.score is not None for r in state.recs)


def test_r_uses_adapter_probabilities_when_it_exists(log):
    # an adapter whose Sad prototype is the feature vector the window holds -> it should pull toward Sad
    x = np.vstack([np.ones((4, 512)), -np.ones((4, 512))])
    adapter = PrototypeAdapter(CLASSES, temperature=0.1).fit(x, ["Sad"] * 4 + ["Happy"] * 4)
    window = FrameWindow(W)
    for _ in range(W):  # generic says Happy 0.5 / Sad 0.4, features look like the user's Sad examples
        p = np.full(7, 0.02)
        p[CLASSES.index("Happy")], p[CLASSES.index("Sad")] = 0.5, 0.38
        window.push(p, np.ones(512))
    ctx = make_ctx(adapters={"prototype": adapter}, choice="prototype")
    state, _, _ = start_recommendation(window, ctx, log)
    assert state.source == "adapter:prototype" and {r.song.emotion for r in state.recs} == {"Sad"}
    generic, src = personalised_probabilities("prototype", {}, np.ones(512), p, 8)
    assert src == "generic" and np.allclose(generic, p)  # no adapter -> generic, same as before


def test_personalised_probabilities_blend_and_fallback():
    x = np.vstack([np.ones((4, 8)), -np.ones((4, 8))])
    adapter = PrototypeAdapter(CLASSES).fit(x, ["Sad"] * 4 + ["Happy"] * 4)
    g = np.full(7, 1 / 7)
    p_user, src = personalised_probabilities("prototype", {"prototype": adapter}, np.ones(8), g, 8)
    p_blend, src2 = personalised_probabilities("prototype+generic", {"prototype": adapter}, np.ones(8), g, 8)
    assert src == "adapter:prototype" and src2 == "adapter:prototype+generic"
    assert np.allclose(p_blend, 0.5 * p_user + 0.5 * g)  # n = 8 = k0 -> w = 0.5
    assert personalised_probabilities("generic", {"prototype": adapter}, np.ones(8), g, 8)[1] == "generic"
    assert personalised_probabilities("logistic", {"prototype": adapter}, np.ones(8), g, 8)[1] == "generic"


def test_rating_flow(log):
    state, _, _ = start_recommendation(filled(W, "Happy"), make_ctx(n=3), log)
    assert handle_rec_key("4", state, log)[3] is False and state.index == 1
    assert handle_rec_key("n", state, log)[3] is False and state.index == 2   # skipped, no rating logged
    msg, _, quit_, closed = handle_rec_key("2", state, log)                   # last song: list closes
    assert closed and not quit_ and "closed" in msg
    _, ratings = load_events(log.jsonl_path)
    assert [(r["song_id"], r["rating"]) for r in ratings] == [(state.recs[0].song.song_id, 4),
                                                              (state.recs[2].song.song_id, 2)]
    assert all(r["recommendation_id"] == state.rec_id for r in ratings)


def test_rec_mode_close_quit_and_ignored_keys(log):
    state, _, _ = start_recommendation(filled(W, "Happy"), make_ctx(), log)
    assert handle_rec_key("6", state, log) == (None, False, False, False)     # emotion keys do nothing here
    assert handle_rec_key(" ", state, log) == (None, False, False, False)
    assert handle_rec_key("R", state, log)[3] is True
    assert handle_rec_key("\x1b", state, log)[3] is True
    assert handle_rec_key("q", state, log)[2] is True
    assert load_events(log.jsonl_path)[1] == []                               # nothing was rated
