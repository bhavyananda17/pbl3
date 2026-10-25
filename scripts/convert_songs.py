"""Build the song library (our CSV schema) from the Kaggle "Spotify tracks" file.

    python scripts/convert_songs.py                       # top 250 English + 250 Hindi by popularity
    python scripts/convert_songs.py --per-language 400

Input:  data/songs/raw/spotify_tracks.csv  (Kaggle gauthamvijayaraj/spotify-tracks-dataset-updated-every-week,
        downloaded once; ~62k tracks with valence and energy from Spotify's audio features; no license is
        stated on its page; it is kept out of git, never commit it)
Output: data/songs/songs.csv  with the columns song_id, title, artist, language, emotion, url, valence, energy.

Choices, all stated so the paper can describe them:
  - only rows tagged English or Hindi in the source's `language` column (the tag is the source's, not verified);
  - rows with valence or energy < 0 (missing) are dropped, duplicates (same title + artist) keep the most popular;
  - the most popular `--per-language` songs per language are kept, so the demo library is recognisable;
  - `emotion` (needed only by the label recommender) is DERIVED: the class whose valence-arousal coordinate in
    configs/recommend.yaml is nearest to the song's (2*valence-1, 2*energy-1). It is not a human label.
  - valence and energy are Spotify's algorithmic estimates, not human annotations.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.base_model import load_config as load_base  # noqa: E402
from src.recommend import load_config as load_rec  # noqa: E402

LANGUAGES = {"English": "en", "Hindi": "hi"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "songs" / "raw" / "spotify_tracks.csv"))
    p.add_argument("--out", default=str(ROOT / "data" / "songs" / "songs.csv"))
    p.add_argument("--per-language", type=int, default=250)
    args = p.parse_args()

    classes = load_base()["classes"]
    coords = np.array([load_rec()["emotion_va"][c] for c in classes])
    with open(args.src, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    kept = []
    for lang_name, lang in LANGUAGES.items():
        best = {}
        for r in rows:
            if r["language"] != lang_name:
                continue
            try:
                v, e, pop = float(r["valence"]), float(r["energy"]), float(r["popularity"])
            except ValueError:
                continue
            if v < 0 or e < 0 or not r["track_id"]:
                continue
            key = (r["track_name"].strip().lower(), r["artist_name"].strip().lower())
            if key not in best or pop > best[key][0]:
                best[key] = (pop, r, v, e)
        top = sorted(best.values(), key=lambda t: (-t[0], t[1]["track_id"]))[: args.per_language]
        for pop, r, v, e in top:
            d = np.hypot(coords[:, 0] - (2 * v - 1), coords[:, 1] - (2 * e - 1))
            kept.append({"song_id": r["track_id"], "title": r["track_name"].strip(), "artist": r["artist_name"].strip(),
                         "language": lang, "emotion": classes[int(np.argmin(d))],
                         "url": r["track_url"].strip(), "valence": f"{v:.3f}", "energy": f"{e:.3f}"})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["song_id", "title", "artist", "language", "emotion", "url", "valence", "energy"])
        w.writeheader()
        w.writerows(kept)
    print(f"Wrote {len(kept)} songs to {args.out}")
    for lang in LANGUAGES.values():
        print(f"  {lang}: " + ", ".join(f"{c} {n}" for c, n in sorted(
            Counter(s["emotion"] for s in kept if s["language"] == lang).items())))


if __name__ == "__main__":
    main()
