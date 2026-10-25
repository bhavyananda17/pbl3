"""Logic behind the Streamlit demo, kept free of Streamlit so it can be tested.

Photos are decoded in memory and never written to disk.
"""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import preprocess

GREEN = (80, 200, 80)


def decode_image(data):
    """Bytes of a JPEG/PNG -> BGR array, or None if it is not an image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def analyse(base, frame_bgr):
    """Detect the largest face and run the frozen model on it.

    Returns {"box", "probs" (7, config class order), "feats" (512,), "annotated" (RGB)} or None if no face
    is found. One photo is one frame, so it is logged with frames_averaged = 1.
    """
    detector = getattr(base, "_detector", None) or preprocess.load_face_detector(base.prep["face_detector"])
    base._detector = detector
    box = preprocess.find_largest_face(detector, frame_bgr, base.prep["face_detector"])
    if box is None:
        return None
    probs, feats = base.predict_face_with_features(preprocess.crop(frame_bgr, box))
    shown = frame_bgr.copy()
    x, y, w, h = box
    cv2.rectangle(shown, (x, y), (x + w, y + h), GREEN, max(2, shown.shape[1] // 200))
    return {"box": box, "probs": probs, "feats": feats, "annotated": cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)}


def va_figure(emotion_va, point, recs=None):
    """Valence-arousal plane: the class anchors, the user's point and the recommended songs (numbered)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#999999", lw=1))
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.axvline(0, color="#cccccc", lw=0.8)
    for name, (v, a) in emotion_va.items():
        ax.scatter([v], [a], s=28, color="#7a7a7a", zorder=2)
        ax.annotate(name, (v, a), textcoords="offset points", xytext=(5, 4), fontsize=8, color="#555555")
    for i, r in enumerate(recs or [], 1):
        sv, se = 2 * r.song.valence - 1, 2 * r.song.energy - 1
        ax.scatter([sv], [se], s=60, color="#0072B2", zorder=3)
        ax.annotate(str(i), (sv, se), textcoords="offset points", xytext=(0, 0), ha="center", va="center",
                    fontsize=7, color="white", zorder=4)
    ax.scatter([point[0]], [point[1]], s=170, marker="*", color="#D55E00", zorder=5, label="your estimate")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("valence (unpleasant → pleasant)")
    ax.set_ylabel("arousal / energy (calm → energetic)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
