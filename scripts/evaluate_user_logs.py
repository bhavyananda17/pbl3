"""Evaluate per-user adapters on correction logs.

    python scripts/evaluate_user_logs.py
    python scripts/evaluate_user_logs.py --log-dir data/corrections --out results/personalization

For each user, trains on the sessions before s and tests on session s, with a learning curve over
the first k training examples. Compares the generic model, the two adapters and each blended with
the generic model. See src/evaluation.py for the protocol and README.md for how to read the output.
Writes results.csv, results.json, sessions.csv and accuracy_vs_k.png to --out.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.base_model import load_config  # noqa: E402
from src.correction_log import DEFAULT_DIR  # noqa: E402
from src.evaluation import METHODS, average_over_users, evaluate, load_logs  # noqa: E402

FOCUS = ["Sad", "Fear", "Disgust"]
STYLE = {"generic": ("#555555", "-"), "logistic": ("#0072B2", "--"), "prototype": ("#D55E00", "--"),
         "logistic+generic": ("#0072B2", "-"), "prototype+generic": ("#D55E00", "-")}


def fmt(x, pct=True):
    return "  n/a" if x is None or np.isnan(x) else (f"{x:5.0%}" if pct else f"{x:5.2f}")


def print_report(rows, avg, session_rows, classes, warnings):
    print("\nCorrection rate per session (share of examples where the user corrected the live model):")
    for r in session_rows:
        print(f"  {r['user']} [{r['model']}] session {r['session']:>2}: {r['n_corrections']}/{r['n_examples']} "
              f"= {r['correction_rate']:.0%}")
    for protocol in ("main", "online"):
        subset = [r for r in rows if r["protocol"] == protocol]
        if not subset:
            continue
        label = ("train on earlier sessions, test on the next" if protocol == "main"
                 else "ONLINE: each example predicted from the same user's earlier examples")
        print(f"\n=== {protocol}: {label} ===")
        head = f"{'user':<6}{'k':>4}  {'method':<18}{'n_test':>6}{'acc':>7}{'macroF1':>8}  " + \
               "  ".join(f"{c[:4]:>5}" for c in classes)
        print(head + "   (per-class recall)")
        for r in subset:
            print(f"{r['user']:<6}{r['k']:>4}  {r['method']:<18}{r['n_test']:>6}{fmt(r['accuracy']):>7}"
                  f"{fmt(r['macro_f1'], False):>8}  " + "  ".join(fmt(r["recall"][c]) for c in classes))
        a = [r for r in avg if r["protocol"] == protocol]
        print(f"--- mean over users ({protocol}) ---")
        for r in a:
            print(f"{'mean':<6}{r['k']:>4}  {r['method']:<18}{r['n_test']:>6}{fmt(r['accuracy']):>7}"
                  f"{fmt(r['macro_f1'], False):>8}  " + "  ".join(fmt(r["recall"][c]) for c in classes)
                  + f"   ({r['n_users']} user(s))")
    if warnings:
        print("\nWARNINGS")
        for w in dict.fromkeys(warnings):
            print(f"  ! {w}")


def save(out, rows, avg, session_rows, classes, warnings):
    out.mkdir(parents=True, exist_ok=True)
    cols = ["user", "model", "protocol", "k", "method", "n_test", "accuracy", "macro_f1"] + \
           [f"recall_{c}" for c in classes]
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["user"], r["model"], r["protocol"], r["k"], r["method"], r["n_test"], r["accuracy"],
                        r["macro_f1"]] + [r["recall"][c] for c in classes])
        for r in avg:
            w.writerow(["MEAN", "", r["protocol"], r["k"], r["method"], r["n_test"], r["accuracy"],
                        r["macro_f1"]] + [r["recall"][c] for c in classes])
    with open(out / "sessions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(session_rows[0]) if session_rows else ["user"])
        w.writeheader()
        w.writerows(session_rows)
    (out / "results.json").write_text(json.dumps(
        {"classes": classes, "per_user": rows, "mean_over_users": avg, "sessions": session_rows,
         "warnings": list(dict.fromkeys(warnings))}, indent=2, default=lambda o: None if o != o else o))


def plot(out, avg, warnings):
    protocols = [p for p in ("main", "online") if any(r["protocol"] == p for r in avg)]
    if not protocols:
        return False
    fig, axes = plt.subplots(1, len(protocols), figsize=(6.5 * len(protocols), 4.6), squeeze=False)
    for ax, protocol in zip(axes[0], protocols):
        ks = sorted({r["k"] for r in avg if r["protocol"] == protocol})
        for m in METHODS:
            ys = {r["k"]: r["accuracy"] for r in avg if r["protocol"] == protocol and r["method"] == m}
            color, ls = STYLE[m]
            ax.plot(range(len(ks)), [ys.get(k, np.nan) for k in ks], color=color, linestyle=ls, marker="o",
                    lw=1.8, label=m)
        n_users = max(r["n_users"] for r in avg if r["protocol"] == protocol)
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("k = the user's labelled examples used for training" if protocol == "main"
                      else "n = earlier examples from the same session (bucket start)")
        ax.set_ylabel("accuracy (mean over users)")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{protocol} protocol, {n_users} user(s)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    if warnings:
        fig.text(0.01, 0.01, "Too little data for the numbers to mean anything: see warnings in the report.",
                 color="#B03030", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "accuracy_vs_k.png", dpi=200)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", default=str(DEFAULT_DIR))
    p.add_argument("--out", default=str(ROOT / "results" / "personalization"))
    args = p.parse_args()
    cfg = load_config()
    classes = cfg["classes"]
    groups = load_logs(args.log_dir, classes)
    if not groups:
        sys.exit(f"No correction logs found in {args.log_dir}. Run a study session first (README).")
    rows, session_rows, warnings = evaluate(groups, classes, cfg)
    avg = average_over_users(rows, classes)
    print_report(rows, avg, session_rows, classes, warnings)
    out = Path(args.out)
    save(out, rows, avg, session_rows, classes, warnings)
    made_plot = plot(out, avg, warnings)
    print(f"\nSaved to {out}: results.csv, results.json, sessions.csv" + (", accuracy_vs_k.png" if made_plot else ""))


if __name__ == "__main__":
    main()
