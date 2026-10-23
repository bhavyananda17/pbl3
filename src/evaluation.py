"""Evaluate per-user adapters on correction logs (used by scripts/evaluate_user_logs.py).

Uses only what the logs hold (the generic model's averaged probabilities, the 512-d features and the
user's labels), so it needs no TensorFlow and no camera.

Main protocol, per user and base model:  for each session s after the first, train on the sessions
before s (chronological order) and test on session s. The learning curve trains on only the first k
examples of that pool. A session is never tested on after being learned from.

Online protocol (only for users with a single session): example i is predicted by adapters trained
on that user's examples 0..i-1 of the same session. It shows a learning curve when there is nothing
earlier to train on; it is reported separately and labelled "online".
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.personalize import blend, make_adapter

METHODS = ["generic", "logistic", "prototype", "logistic+generic", "prototype+generic"]
ONLINE_BUCKETS = [0, 1, 2, 4, 8, 16]  # an example with n earlier examples falls in the largest bucket <= n


# -- loading -------------------------------------------------------------------------------------
def load_logs(log_dir, classes):
    """{(user_id, model, sha256, feature_layer): [session dict, ...]} sorted by session id.

    A session dict has: session, records, features (n, d), generic (n, 7) in `classes` order,
    labels (list of names). Sessions whose class order differs from `classes` are rejected.
    """
    groups = defaultdict(list)
    for jsonl in sorted(Path(log_dir).glob("*/s*.jsonl")):
        records = [json.loads(l) for l in open(jsonl) if l.strip()]
        if not records:
            continue
        feats = np.load(jsonl.with_suffix(".npy"))
        r0 = records[0]
        if r0["class_order"] != list(classes):
            raise ValueError(f"{jsonl}: class order {r0['class_order']} differs from configs/base.yaml")
        rows = [r["feature_row"] for r in records]
        key = (r0["user_id"], r0["model"], r0["model_sha256"], r0["feature_layer"])
        groups[key].append({
            "session": r0["session_id"], "path": str(jsonl), "records": records,
            "features": feats[rows].astype(np.float64),
            "generic": np.array([r["probabilities"] for r in records], dtype=np.float64),
            "labels": [r["user_label"] for r in records],
        })
    return {k: sorted(v, key=lambda s: s["session"]) for k, v in groups.items()}


def pool(sessions):
    """Concatenate sessions in chronological order -> (features, labels)."""
    if not sessions:
        return np.zeros((0, 0)), []
    return np.vstack([s["features"] for s in sessions]), [l for s in sessions for l in s["labels"]]


def main_splits(sessions):
    """[(test_session, training_sessions)] with training = every earlier session. Never the test one."""
    return [(sessions[i], sessions[:i]) for i in range(1, len(sessions))]


def resolve_ks(ks, pool_size):
    """Config ks ('all' allowed) -> sorted unique prefix sizes that fit in the pool."""
    out = {pool_size if k == "all" else int(k) for k in ks}
    return sorted(k for k in out if k <= pool_size)


# -- prediction and metrics ----------------------------------------------------------------------
def predict_methods(train_x, train_y, test_x, test_generic, classes, pcfg):
    """{method: (N, 7) probabilities} for adapters fitted on the first len(train_y) examples."""
    n = len(train_y)
    out = {"generic": test_generic}
    for kind in ("logistic", "prototype"):
        adapter = make_adapter(kind, classes, pcfg).fit(train_x[:n], train_y)
        p_user = adapter.predict_proba(test_x, test_generic)
        out[kind] = p_user
        out[kind + "+generic"] = blend(p_user, test_generic, n, pcfg["k0"])
    return out


def score(y_true, y_pred, classes):
    """accuracy, macro-F1 (over classes present in truth or predictions) and per-class recall
    (NaN for a class with no true examples)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    recall = {}
    f1s = []
    for c in range(len(classes)):
        t, p = y_true == c, y_pred == c
        recall[classes[c]] = float((t & p).sum() / t.sum()) if t.any() else float("nan")
        if t.any() or p.any():
            tp = (t & p).sum()
            prec = tp / p.sum() if p.any() else 0.0
            rec = tp / t.sum() if t.any() else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"n_test": int(n), "accuracy": float((y_true == y_pred).mean()) if n else float("nan"),
            "macro_f1": float(np.mean(f1s)) if f1s else float("nan"), "recall": recall}


class Collector:
    """Pools (truth, prediction) per (user, model, protocol, k, method) before scoring."""

    def __init__(self, classes):
        self.classes = classes
        self.data = defaultdict(lambda: ([], []))

    def add(self, key, y_true_idx, probs_by_method):
        for m, p in probs_by_method.items():
            t, pr = self.data[key + (m,)]
            t.extend(y_true_idx)
            pr.extend(np.asarray(p).argmax(1).tolist())

    def rows(self):
        out = []
        for (user, model, protocol, k, method), (t, p) in sorted(self.data.items(), key=lambda kv: kv[0]):
            out.append({"user": user, "model": model, "protocol": protocol, "k": k, "method": method,
                        **score(t, p, self.classes)})
        return out


# -- evaluation ----------------------------------------------------------------------------------
def evaluate(groups, classes, cfg):
    """Returns (rows, session_rows, warnings). rows: one per (user, model, protocol, k, method)."""
    pcfg = cfg["personalize"]
    col, session_rows, warnings = Collector(classes), [], []
    for (user, model, sha, layer), sessions in groups.items():
        tag = f"{user} [{model}]"
        n_total = sum(len(s["labels"]) for s in sessions)
        counts = defaultdict(int)
        for s in sessions:
            for l in s["labels"]:
                counts[l] += 1
        for s in sessions:
            corr = sum(r["source"] == "correct" for r in s["records"])
            session_rows.append({"user": user, "model": model, "session": s["session"],
                                 "n_examples": len(s["labels"]), "n_corrections": corr,
                                 "correction_rate": corr / len(s["labels"])})
        if n_total < pcfg["min_examples_warn"]:
            warnings.append(f"{tag}: only {n_total} labelled examples in total (< {pcfg['min_examples_warn']}); "
                            "numbers are anecdotal.")
        missing = [c for c in classes if c not in counts]
        if missing:
            warnings.append(f"{tag}: no examples at all for {', '.join(missing)}; recall for them is undefined "
                            "and adapters fall back to the generic model for them.")
        if len(sessions) < 2:
            warnings.append(f"{tag}: only {len(sessions)} session; the train-on-earlier / test-on-later "
                            "protocol needs at least 2. Only the 'online' protocol is reported.")
            _online(col, user, model, sessions[0], classes, pcfg)
            continue
        for test, train in main_splits(sessions):
            px, py = pool(train)
            tx, ty = test["features"], np.array([classes.index(l) for l in test["labels"]])
            if len(ty) < pcfg["min_test_examples_warn"]:
                warnings.append(f"{tag}: test session {test['session']} has only {len(ty)} examples "
                                f"(< {pcfg['min_test_examples_warn']}).")
            for k in resolve_ks(pcfg["learning_curve_ks"], len(py)):
                col.add((user, model, "main", k), ty,
                        predict_methods(px[:k], py[:k], tx, test["generic"], classes, pcfg))
    rows = col.rows()
    n_users = len({r["user"] for r in rows})
    if n_users < 3:
        warnings.append(f"Averages are over {n_users} user(s); they say nothing about users in general.")
    return rows, session_rows, warnings


def _online(col, user, model, session, classes, pcfg):
    x, labels, generic = session["features"], session["labels"], session["generic"]
    y = np.array([classes.index(l) for l in labels])
    for i in range(len(y)):
        bucket = max(b for b in ONLINE_BUCKETS if b <= i)
        probs = predict_methods(x[:i], labels[:i], x[i:i + 1], generic[i:i + 1], classes, pcfg)
        col.add((user, model, "online", bucket), y[i:i + 1], probs)


def average_over_users(rows, classes):
    """Unweighted mean over users of accuracy, macro-F1 and per-class recall, per (protocol, k, method)."""
    by = defaultdict(list)
    for r in rows:
        by[(r["protocol"], r["k"], r["method"])].append(r)
    out = []
    for (protocol, k, method), rs in sorted(by.items()):
        recall = {c: (float(np.nanmean([r["recall"][c] for r in rs]))
                      if any(not np.isnan(r["recall"][c]) for r in rs) else float("nan")) for c in classes}
        out.append({"protocol": protocol, "k": k, "method": method, "n_users": len(rs),
                    "n_test": sum(r["n_test"] for r in rs),
                    "accuracy": float(np.nanmean([r["accuracy"] for r in rs])),
                    "macro_f1": float(np.nanmean([r["macro_f1"] for r in rs])), "recall": recall})
    return out
