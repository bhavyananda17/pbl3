"""Study app: confirm or correct the detected emotion, logging each example for per-user training.

    python -m src.study_app --user-id P01
    python -m src.study_app --user-id P01 --model fer2013_clean --camera 1
    python -m src.study_app --list-cameras

Keys:  1-7 = the correct emotion (map shown on screen)   Space = the prediction is correct
       S = skip   Q = quit   R = recommend songs
After R the song list is shown: 1-5 rates the highlighted song (then moves on), N = next song without
rating, R or Esc = close the list. Recommendations use the user's saved adapter if there is one
(scripts/fit_user_adapter.py), otherwise the generic model, and are logged in the session file.
A keypress logs the probabilities and features averaged over the last W frames with a face
(W = study.correction_window in configs/base.yaml). With fewer than W such frames nothing is
logged and a warning is shown. No images or frames are ever saved (see src/correction_log.py).
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

from src import personalize, preprocess, recommend
from src.base_model import BaseModel
from src.correction_log import DEFAULT_DIR, CorrectionLog, check_user_id, next_session

GREEN, GREY, WHITE, DARK, AMBER = (80, 200, 80), (160, 160, 160), (255, 255, 255), (30, 30, 30), (0, 190, 255)
PANEL_W = 300
WINDOW = "Study app (Q to quit)"
MESSAGE_SECONDS = 2.5


class FrameWindow:
    """The last `size` frames that had a face: their probabilities and features."""

    def __init__(self, size):
        self.size = size
        self._probs, self._feats = deque(maxlen=size), deque(maxlen=size)

    def push(self, probs, feats):
        self._probs.append(np.asarray(probs, dtype=np.float64))
        self._feats.append(np.asarray(feats, dtype=np.float64))

    def clear(self):
        self._probs.clear()
        self._feats.clear()

    def __len__(self):
        return len(self._probs)

    @property
    def ready(self):
        return len(self) >= self.size

    def mean(self):
        """(mean probabilities, mean features, frames averaged); None if empty."""
        if not self._probs:
            return None
        return np.mean(self._probs, axis=0), np.mean(self._feats, axis=0).astype(np.float32), len(self)


def key_map(classes):
    """{'1': 'Angry', ...} from the config's class order."""
    return {str(i + 1): c for i, c in enumerate(classes)}


def handle_key(ch, window, classes, log):
    """Apply one keypress. Returns (message, is_warning, quit)."""
    if ch in ("q", "Q"):
        return "Quitting", False, True
    if ch in ("s", "S"):
        return "Skipped", False, False
    keys = key_map(classes)
    if ch != " " and ch not in keys:
        return None, False, False
    if not window.ready:
        return f"Not logged: {len(window)}/{window.size} face frames, hold still", True, False
    probs, feats, n = window.mean()
    label = classes[int(np.argmax(probs))] if ch == " " else keys[ch]
    rec = log.add(probs, feats, label, n)
    window.clear()  # the next example needs fresh frames
    verb = "Confirmed" if rec["source"] == "confirm" else f"Corrected {rec['predicted_label']} ->"
    return f"{verb} {label} (#{log.counts['confirm'] + log.counts['correct']})", False, False


class RecContext:
    """What key R needs: the recommender, the user's adapters, and the songs shown recently."""

    def __init__(self, recommender, classes, adapters, choice, k0, n, language, recent_memory):
        self.recommender, self.classes, self.adapters = recommender, classes, adapters
        self.choice, self.k0, self.n, self.language = choice, k0, n, language
        self.recent = deque(maxlen=recent_memory)


class RecState:
    """The song list on screen after R: which song is highlighted and what was rated."""

    def __init__(self, rec_id, recs, source):
        self.rec_id, self.recs, self.source = rec_id, recs, source
        self.index, self.ratings = 0, {}

    @property
    def current(self):
        return self.recs[self.index].song


def start_recommendation(window, ctx, log):
    """Key R. Returns (RecState or None, message, is_warning). Needs a full window of face frames."""
    if not window.ready:
        return None, f"No songs: {len(window)}/{window.size} face frames, hold still", True
    probs, feats, _ = window.mean()
    p, source = personalize.personalised_probabilities(ctx.choice, ctx.adapters, feats, probs, ctx.k0)
    recs = ctx.recommender.recommend(p, n=ctx.n, language=ctx.language, exclude=ctx.recent)
    if not recs:
        return None, "No songs match (language or recently shown)", True
    rec_id = log.add_recommendation(ctx.recommender.kind, p, source, ctx.language, recs)
    ctx.recent.extend(r.song.song_id for r in recs)
    top = ctx.classes[int(np.argmax(p))]
    print(f"Recommendations for {top} ({ctx.recommender.kind}, {source}):")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r.song.title} - {r.song.artist} [{r.song.language}] {r.song.url}")
    return RecState(rec_id, recs, source), f"Songs for {top} ({source})", False


def handle_rec_key(ch, state, log):
    """Keypress while the song list is shown. Returns (message, is_warning, quit, list_closed)."""
    if ch in ("q", "Q"):
        return "Quitting", False, True, True
    if ch in ("r", "R", "\x1b"):
        return "Song list closed", False, False, True
    if ch in ("n", "N", "1", "2", "3", "4", "5"):
        msg = None
        if ch != "n" and ch != "N":
            state.ratings[state.index] = int(ch)
            log.add_rating(state.rec_id, state.current.song_id, int(ch))
            msg = f"Rated {state.current.title[:20]} {ch}/5"
        state.index += 1
        if state.index >= len(state.recs):
            return (msg or "End of list") + " - list closed", False, False, True
        return msg or "Next song", False, False, False
    return None, False, False, False


def draw_panel(height, base, window, log, args, message, warning, face_found, rec_state=None):
    panel = np.full((height, PANEL_W, 3), 40, dtype=np.uint8)
    put = lambda text, y, color=WHITE, scale=0.5, thick=1: cv2.putText(
        panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    classes = base.classes
    put(f"user {args.user_id}  session {log.session}", 20, GREY)
    put(f"model {base.name}", 38, GREY)
    avg = window.mean()
    if avg is not None:
        probs = avg[0]
        best = int(np.argmax(probs))
        put(f"{classes[best]} {probs[best]:.0%}", 72, GREEN, 0.9, 2)
    else:
        probs, best = np.zeros(len(classes)), -1
        put("No face found" if not face_found else "...", 72, AMBER, 0.7, 2)
    put(f"face frames {len(window)}/{window.size}", 92, GREEN if window.ready else AMBER)
    if rec_state is not None:
        put(f"Songs ({rec_state.source})", 114, AMBER)
        for i, r in enumerate(rec_state.recs):
            y = 138 + i * 36
            color = GREEN if i == rec_state.index else WHITE
            rating = f"  {rec_state.ratings[i]}/5" if i in rec_state.ratings else ""
            put(f"{'>' if i == rec_state.index else ' '} {i + 1}. {r.song.title[:22]}{rating}", y, color, 0.45)
            put(f"     {r.song.artist[:26]} [{r.song.language}]", y + 15, GREY, 0.4)
        put("1-5 = rate   N = next", 138 + 5 * 36 + 10, GREY)
        put("R / Esc = close list", 138 + 5 * 36 + 28, GREY)
    else:
        for i, name in enumerate(classes):
            y = 118 + i * 24
            put(f"{i + 1} {name}", y, WHITE)
            cv2.rectangle(panel, (120, y - 11), (120 + int(probs[i] * 160), y + 3), GREEN if i == best else GREY, -1)
        put("Keys: 1-7 = correct emotion", 118 + 7 * 24 + 14, GREY)
        put("Space = prediction is right", 118 + 7 * 24 + 32, GREY)
        put("S = skip  R = songs  Q = quit", 118 + 7 * 24 + 50, GREY)
    put(f"confirmed {log.counts['confirm']}   corrected {log.counts['correct']}", height - 50, WHITE, 0.55)
    if message:
        put(message[:34], height - 28, AMBER if warning else GREEN)
        if len(message) > 34:
            put(message[34:68], height - 12, AMBER if warning else GREEN)
    return panel


def list_cameras(max_index=4):
    """Print which camera indexes open (no pictures are saved). Use the printed size to tell them apart."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        print(f"index {i}: " + (f"{frame.shape[1]}x{frame.shape[0]}" if ok else "no camera"))
    print("On a Mac with an iPhone nearby, Continuity Camera can be index 0; try --camera 1.")


def run(args, scripted_keys=None):
    """Main loop. `scripted_keys` (a list of characters, one per frame, None = no key) replaces the
    keyboard and hides the window; it is for tests."""
    for w in check_user_id(args.user_id):
        print(f"WARNING: {w}")
    base = BaseModel(args.model)
    session = args.session if args.session is not None else next_session(args.log_dir, args.user_id)
    study = base.cfg["study"]
    m = base.cfg["models"][base.name]
    log = CorrectionLog(args.user_id, session, base.name, m["sha256"], base.classes, base.feature_layer,
                        base.feature_dim, args.log_dir)
    window = FrameWindow(study["correction_window"])
    print(log.notice(window.size))

    rcfg = recommend.load_config()
    try:
        recommender = recommend.make_recommender(args.recommender or rcfg["recommender"], rcfg, base.classes,
                                                 args.songs)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Could not load the song list: {e}")
    adapters = personalize.load_user_adapters(args.user_id, base.name, m["sha256"], base.classes)
    ctx = RecContext(recommender, base.classes, adapters, rcfg["probabilities"], base.cfg["personalize"]["k0"],
                     rcfg["n"], args.language, rcfg["recent_memory"])
    print(f"Key R: {recommender.kind} recommender, {len(recommender.songs)} songs"
          + (f", language {args.language}" if args.language else "") + ". Probabilities: "
          + (f"saved adapter(s) {sorted(adapters)} ({rcfg['probabilities']})" if adapters else "generic (no saved adapter)"))

    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. On macOS, allow your terminal app under System Settings > "
                         "Privacy & Security > Camera. Try --list-cameras.")
    detector = preprocess.load_face_detector(base.prep["face_detector"])
    headless = scripted_keys is not None
    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.moveWindow(WINDOW, 40, 40)
        try:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        print(f"Window '{WINDOW}' opened. Press Q in it to quit.")
    live = isinstance(source, int)
    message, warning, message_until, failed = "", False, 0.0, 0
    frame_i, quit_, rec_state = 0, False, None
    while not quit_:
        ok, frame = cap.read()
        if not ok:
            failed += 1
            if live and failed < 50:
                time.sleep(0.1)
                continue
            break
        failed = 0
        if live:
            frame = cv2.flip(frame, 1)
        box = preprocess.find_largest_face(detector, frame, base.prep["face_detector"])
        if box is not None:
            window.push(*base.predict_face_with_features(preprocess.crop(frame, box)))
        else:
            window.clear()  # a new person must not mix with old frames
        if not headless:
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
            shown = frame if frame.shape[0] >= 480 else cv2.resize(frame, None, fx=480 / frame.shape[0],
                                                                    fy=480 / frame.shape[0])
            panel = draw_panel(shown.shape[0], base, window, log, args,
                               message if time.time() < message_until else "", warning, box is not None,
                               rec_state)
            cv2.imshow(WINDOW, np.hstack([shown, panel]))
            k = cv2.waitKey(1) & 0xFF
            ch = chr(k) if 0 < k < 128 else None
        else:
            ch = scripted_keys[frame_i] if frame_i < len(scripted_keys) else "q"
        frame_i += 1
        if ch:
            if rec_state is not None:
                msg, warn, quit_, closed = handle_rec_key(ch, rec_state, log)
                rec_state = None if closed else rec_state
            elif ch in ("r", "R"):
                rec_state, msg, warn = start_recommendation(window, ctx, log)
            else:
                msg, warn, quit_ = handle_key(ch, window, base.classes, log)
            if msg:
                message, warning, message_until = msg, warn, time.time() + MESSAGE_SECONDS
                print(("WARNING: " if warn else "") + msg)
    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    n = log.counts["confirm"] + log.counts["correct"]
    print(f"\nSession {session} done: {n} examples ({log.counts['confirm']} confirmed, "
          f"{log.counts['correct']} corrected), {log.n_recommendations} song list(s) shown.")
    if n or log.n_recommendations:
        print(f"Saved: {log.jsonl_path}" + (f"\n       {log.npy_path}" if n else ""))
    else:
        print("Nothing was logged, so no files were created.")
    return log


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user-id", help="pseudonym such as P01 (required)")
    p.add_argument("--session", type=int, help="session number (default: next unused for this user)")
    p.add_argument("--model", default=None, help="rafdb_clean (default) or fer2013_clean")
    p.add_argument("--log-dir", default=str(DEFAULT_DIR), help="where logs go (default data/corrections/)")
    p.add_argument("--camera", default="0", help="camera index (0 = default) or a video file")
    p.add_argument("--recommender", choices=["label", "va"], help="recommender for key R (default: configs/recommend.yaml)")
    p.add_argument("--songs", help="songs CSV (default: songs_csv in configs/recommend.yaml)")
    p.add_argument("--language", help="only recommend songs in this language (as in the CSV, e.g. en)")
    p.add_argument("--list-cameras", action="store_true", help="show which camera indexes work and exit")
    args = p.parse_args()
    if args.list_cameras:
        list_cameras()
        return
    if not args.user_id:
        p.error("--user-id is required")
    try:
        check_user_id(args.user_id)
    except ValueError as e:
        p.error(str(e))
    run(args)


if __name__ == "__main__":
    main()
