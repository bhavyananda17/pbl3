"""Front-end demo: photo -> expression estimate -> confirm or correct -> adapter -> songs.

    streamlit run src/streamlit_app.py

Runs locally, no network. A photo is analysed in memory and is never saved; only probabilities, a
512-number feature vector and labels are logged (same files as src/study_app.py).
"""

import os
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import demo, personalize, recommend  # noqa: E402
from src.base_model import BaseModel, load_config  # noqa: E402
from src.correction_log import DEFAULT_DIR, CorrectionLog, check_user_id, next_session  # noqa: E402
from src.evaluation import fit_user_adapters  # noqa: E402

st.set_page_config(page_title="Personalised expression + music demo", layout="wide")

LOG_DIR = Path(os.environ.get("PBL_LOG_DIR", DEFAULT_DIR))  # override for tests only
CFG = load_config()
RCFG = recommend.load_config()
CLASSES = CFG["classes"]
LANGS = {"Any": None, "English": "en", "Hindi": "hi"}


@st.cache_resource(show_spinner="Loading the frozen CNN ...")
def get_base(name):
    return BaseModel(name)


@st.cache_resource
def get_recommender(kind):
    return recommend.make_recommender(kind, RCFG, CLASSES)


def get_adapters(user_id, model_name):
    sha = CFG["models"][model_name]["sha256"]
    return personalize.load_user_adapters(user_id, model_name, sha, CLASSES)


def ensure_log(user_id, model_name):
    """The session log for this browser session; created on first use (never while the id is being typed)."""
    key = (user_id, model_name)
    if st.session_state.get("log_key") != key:
        st.session_state.log = CorrectionLog(user_id, next_session(LOG_DIR, user_id), model_name,
                                             CFG["models"][model_name]["sha256"], CLASSES, CFG["features"]["layer"],
                                             CFG["features"]["dim"], LOG_DIR)
        st.session_state.log_key = key
        st.session_state.recent = deque(maxlen=RCFG["recent_memory"])
    return st.session_state.log


# ---- sidebar ---------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Session")
    user_id = st.text_input("Participant ID (pseudonym, e.g. P01)", value="P01").strip()
    model_name = st.selectbox("Base model", list(CFG["models"]), index=list(CFG["models"]).index(CFG["default_model"]))
    kind = st.selectbox("Recommender", ["va", "label"], index=["va", "label"].index(RCFG["recommender"]),
                        help="va: nearest songs in valence-arousal space. label: songs labelled with the top emotion.")
    language = LANGS[st.selectbox("Song language", list(LANGS))]
    try:
        warnings_ = check_user_id(user_id)
        id_ok = True
    except ValueError as e:
        st.error(str(e))
        warnings_, id_ok = [], False
    for w in warnings_:
        st.warning(w)
    st.caption("Privacy: the photo is analysed in memory and never saved. Only probabilities, a feature vector and "
               "your labels are stored, under data/corrections/ on this computer.")
    st.caption("The model estimates the *displayed expression*, not how you feel.")

if not id_ok:
    st.stop()

base = get_base(model_name)
recommender = get_recommender(kind)
adapters = get_adapters(user_id, model_name)
choice = RCFG["probabilities"]

with st.sidebar:
    st.subheader("Your adapter")
    if adapters:
        a = adapters[min(adapters, key=lambda k: k != "logistic")]
        st.success(f"Active: {RCFG['probabilities']}, fitted on {a.n_examples} labelled examples "
                   f"({', '.join(f'{c} {n}' for c, n in a.counts.items())})")
    else:
        st.info("No adapter yet: using the generic model. Log some corrections, then refit.")
    if st.button("Refit my adapter from my logs"):
        fitted = fit_user_adapters(user_id, model_name, CFG, LOG_DIR)
        if not fitted:
            st.warning("No logged examples for this participant and model yet.")
        else:
            n, counts = fitted["logistic"]
            st.success(f"Fitted on {n} examples ({', '.join(f'{c} {k}' for c, k in counts.items())}).")
            if len(counts) < 2:
                st.warning("Only one class seen so far: the adapter does nothing until at least 2 are labelled.")
            st.rerun()
    if len(recommender.songs) and recommender.songs[0].title.startswith("EXAMPLE"):
        st.warning("Song library: fake EXAMPLE rows (data/songs/songs.csv not found).")
    else:
        st.caption(f"Song library: {len(recommender.songs)} songs (valence/energy from Spotify's audio features).")

# ---- main ------------------------------------------------------------------------------------------------
st.title("Personalised facial-expression recognition and music recommendation")
st.write("Take a photo showing an expression. The frozen CNN estimates it, you confirm or correct it, your "
         "adapter learns from the corrections, and songs are picked from the valence-arousal point.")

left, right = st.columns([1, 1])
with left:
    cam_tab, up_tab = st.tabs(["Camera", "Upload a photo"])
    with cam_tab:
        shot = st.camera_input("Look at the camera and take a photo")
    with up_tab:
        upload = st.file_uploader("...or upload a face photo (if the camera is unavailable)",
                                  type=["jpg", "jpeg", "png"])
data = None
for src in (shot, upload):
    if src is not None:
        data = src.getvalue()
if data is None and os.environ.get("PBL_DEMO_IMAGE"):  # test hook: pre-load an image without the browser camera
    data = Path(os.environ["PBL_DEMO_IMAGE"]).read_bytes()
if data is None:
    with right:
        st.info("Waiting for a photo.")
    st.stop()

photo_id = hash(data)
if st.session_state.get("photo_id") != photo_id:  # new photo: forget the previous songs and feedback
    st.session_state.photo_id, st.session_state.logged, st.session_state.recs = photo_id, None, None
frame = demo.decode_image(data)
result = demo.analyse(base, frame) if frame is not None else None
if result is None:
    with right:
        st.error("No face found. Face the camera in good light and try again.")
    st.stop()

generic = result["probs"]
used, source = personalize.personalised_probabilities(choice, adapters, result["feats"], generic, CFG["personalize"]["k0"])
shown_label = CLASSES[int(np.argmax(used))]

with right:
    st.subheader(f"Detected expression: {shown_label} ({used.max():.0%})")
    st.caption(f"Probabilities from: {'your personal adapter blended with the generic model' if source != 'generic' else 'the generic model'}"
               + (f". Generic model alone says {CLASSES[int(np.argmax(generic))]} ({generic.max():.0%})."
                  if source != "generic" else "."))
    img_col, bar_col = st.columns([1, 2])
    img_col.image(result["annotated"], caption="Detected face (not saved)", use_container_width=True)
    frame_df = pd.DataFrame({"Generic": generic, "Used": used}, index=CLASSES) if source != "generic" \
        else pd.DataFrame({"Generic": generic}, index=CLASSES)
    bar_col.bar_chart(frame_df, horizontal=True, height=260)

# ---- feedback --------------------------------------------------------------------------------------------
st.subheader("Is that right?")


def log_label(label):
    log = ensure_log(user_id, model_name)
    rec = log.add(generic, result["feats"], label, 1)
    st.session_state.logged = (label, rec["source"], log.counts["confirm"] + log.counts["correct"])


if st.session_state.logged:
    label, src_, n = st.session_state.logged
    st.success(f"Logged: {label} ({'confirmed' if src_ == 'confirm' else 'corrected'}). "
               f"{n} example(s) this session. Take a new photo for the next one.")
else:
    cols = st.columns(len(CLASSES) + 1)
    if cols[0].button(f"✅ Yes, {shown_label}", type="primary"):
        log_label(shown_label)
        st.rerun()
    for col, name in zip(cols[1:], CLASSES):
        if col.button(f"It was {name}", key=f"fix_{name}"):
            log_label(name)
            st.rerun()

# ---- songs -----------------------------------------------------------------------------------------------
st.subheader("Songs for this expression")
target = None
if kind == "va" and hasattr(recommender, "target"):
    target = recommender.target(used)
    st.caption(f"Valence-arousal estimate: valence {target[0]:+.2f}, arousal {target[1]:+.2f}")

if st.button("Recommend 5 songs"):
    recs = recommender.recommend(used, n=RCFG["n"], language=language, exclude=st.session_state.get("recent", ()))
    if not recs:
        st.warning("No songs match this language or all were shown recently.")
    else:
        log = ensure_log(user_id, model_name)
        rec_id = log.add_recommendation(recommender.kind, used, source if source == "generic" else f"adapter:{choice}",
                                        language, recs)
        st.session_state.recent.extend(r.song.song_id for r in recs)
        st.session_state.recs = (rec_id, recs, set())

if st.session_state.get("recs"):
    rec_id, recs, rated = st.session_state.recs
    cols = st.columns([3, 2])
    with cols[0]:
        for i, r in enumerate(recs):
            s = r.song
            a, b = st.columns([4, 2])
            a.markdown(f"**{i + 1}. [{s.title}]({s.url})**  \n{s.artist} · {s.language}")
            stars = b.feedback("stars", key=f"stars_{rec_id}_{i}")
            if stars is not None and (rec_id, i) not in st.session_state.get("rated_keys", set()):
                ensure_log(user_id, model_name).add_rating(rec_id, s.song_id, stars + 1)
                st.session_state.setdefault("rated_keys", set()).add((rec_id, i))
        st.caption("Rate how well each song fits your mood (1-5 stars). Ratings are logged.")
    with cols[1]:
        if target is not None:
            st.pyplot(demo.va_figure(RCFG["emotion_va"], target, recs))
