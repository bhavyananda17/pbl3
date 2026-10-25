"""Streamlit demo: photo analysis and a headless run of the app.  Run: pytest"""

import glob
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src import demo
from src.base_model import BaseModel, load_config
from src.recommend import load_config as load_rec

ROOT = Path(__file__).resolve().parent.parent
OLD_FACES = sorted(glob.glob(str(ROOT.parent / "facial-emotion-recognition-cnn" / "data" / "rafdb" / "raw" /
                                 "DATASET" / "test" / "4" / "*.jpg")))


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def face_photo(tmp_path):
    """A real RAF-DB face on a plain background, written to a temp file (never inside the repo)."""
    face = cv2.resize(cv2.imread(OLD_FACES[3]), (300, 300), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((480, 640, 3), 200, np.uint8)
    canvas[90:390, 170:470] = face
    path = tmp_path / "photo.png"
    path.write_bytes(png_bytes(canvas))
    return path


@pytest.fixture(scope="module")
def base():
    return BaseModel()


def test_decode_image_rejects_non_images():
    assert demo.decode_image(b"not an image") is None
    assert demo.decode_image(png_bytes(np.zeros((8, 8, 3), np.uint8))).shape == (8, 8, 3)


def test_no_face_returns_none(base):
    assert demo.analyse(base, np.full((480, 640, 3), 128, np.uint8)) is None


@pytest.mark.skipif(not OLD_FACES, reason="old repo's RAF-DB data not available")
def test_face_gives_probabilities_features_and_annotated_copy(base, tmp_path):
    frame = demo.decode_image(face_photo(tmp_path).read_bytes())
    r = demo.analyse(base, frame)
    assert r["probs"].shape == (7,) and np.isclose(r["probs"].sum(), 1.0, atol=1e-4)
    assert r["feats"].shape == (512,) and r["annotated"].shape == frame.shape


def test_va_figure_builds():
    from src.recommend import Recommendation, Song

    rec = Recommendation(Song("a", "t", "ar", "en", "Happy", "u", 0.8, 0.7), 0.1)
    fig = demo.va_figure(load_rec()["emotion_va"], (0.2, 0.1), [rec])
    assert fig.axes and len(fig.axes[0].collections) >= 8


def run_app(monkeypatch, tmp_path, photo=None):
    monkeypatch.setenv("PBL_LOG_DIR", str(tmp_path / "logs"))
    if photo:
        monkeypatch.setenv("PBL_DEMO_IMAGE", str(photo))
    else:
        monkeypatch.delenv("PBL_DEMO_IMAGE", raising=False)
    return AppTest.from_file(str(ROOT / "src" / "streamlit_app.py"), default_timeout=180).run()


def test_app_waits_for_a_photo(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path)
    assert not at.exception and any("Waiting for a photo" in i.value for i in at.info)
    assert not (tmp_path / "logs").exists()  # nothing is created until the user logs something


@pytest.mark.skipif(not OLD_FACES, reason="old repo's RAF-DB data not available")
def test_app_confirm_recommend_and_log(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path, face_photo(tmp_path))
    assert not at.exception
    confirm = [b for b in at.button if b.label.startswith("✅")][0]
    confirm.click().run()
    assert not at.exception and any("Logged" in s.value for s in at.success)
    [b for b in at.button if b.label == "Recommend 5 songs"][0].click().run()
    assert not at.exception
    from src.correction_log import load_events, load_session

    jsonl = next((tmp_path / "logs").rglob("*.jsonl"))
    records, feats = load_session(jsonl)
    recs, _ = load_events(jsonl)
    assert len(records) == 1 and records[0]["frames_averaged"] == 1 and feats.shape == (1, 512)
    assert len(recs) == 1 and len(recs[0]["songs"]) == 5 and recs[0]["recommender"] == "va"
    assert sorted(p.suffix for p in (tmp_path / "logs").rglob("*") if p.is_file()) == [".jsonl", ".npy"]  # no images
