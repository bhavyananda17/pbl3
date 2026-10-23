"""Fit and save a user's adapters from all their correction logs, for key R in the study app.

    python scripts/fit_user_adapter.py --user-id P01
    python scripts/fit_user_adapter.py --user-id P01 --model fer2013_clean

Uses every session of the user for the chosen base model and writes
models/users/<user_id>/<model>_logistic.* and <model>_prototype.* (gitignored). Do not evaluate on
sessions you fit on: use scripts/evaluate_user_logs.py for that.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.base_model import load_config  # noqa: E402
from src.correction_log import DEFAULT_DIR  # noqa: E402
from src.evaluation import load_logs, pool  # noqa: E402
from src.personalize import KINDS, make_adapter  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user-id", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--log-dir", default=str(DEFAULT_DIR))
    args = p.parse_args()
    cfg = load_config()
    model = args.model or cfg["default_model"]
    classes = cfg["classes"]
    sessions = [s for (user, m, _, _), ss in load_logs(args.log_dir, classes).items()
                if user == args.user_id and m == model for s in ss]
    if not sessions:
        sys.exit(f"No correction logs for {args.user_id} with model {model} in {args.log_dir}.")
    x, y = pool(sorted(sessions, key=lambda s: s["session"]))
    sha = cfg["models"][model]["sha256"]
    for kind in KINDS:
        adapter = make_adapter(kind, classes, cfg["personalize"]).fit(x, y)
        stem = adapter.save(args.user_id, model, sha)
        print(f"{kind}: fitted on {adapter.n_examples} examples ({adapter.counts}) -> {stem}.npz")


if __name__ == "__main__":
    main()
