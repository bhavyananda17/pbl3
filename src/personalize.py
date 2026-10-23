"""Per-user adapters that learn from a user's corrections, on top of the frozen base model.

    adapter = LogisticAdapter(classes, C=0.1).fit(features, labels)   # features (n, 512), labels = names
    p_user = adapter.predict_proba(features, generic_probs)           # (n, 7), config class order
    p = blend(p_user, generic_probs, n=adapter.n_examples, k0=8)

The base model is never touched: adapters only see its 512-d features (batch_normalization_8) and
its probabilities.

Classes the user has not labelled yet: an adapter models only the classes it has seen. Its output
over those classes is rescaled to the generic model's total probability on them, and unseen classes
keep their generic probability, so the result is still a distribution and an unseen class is never
pushed to zero. With fewer than two seen classes the logistic adapter cannot be fitted and returns
the generic probabilities unchanged (a prototype adapter with one class gives the same result).

Fitted adapters are saved under models/users/<user_id>/ as .npz + .json (no pickle).
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
USERS_DIR = ROOT / "models" / "users"


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _l2_normalise(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def blend(p_user, p_generic, n, k0):
    """p = w * p_user + (1 - w) * p_generic with w = n / (n + k0); n = the user's labelled examples."""
    w = n / (n + k0) if n > 0 else 0.0
    return w * np.asarray(p_user) + (1.0 - w) * np.asarray(p_generic)


class Adapter:
    kind = "base"

    def __init__(self, classes):
        self.classes = list(classes)
        self.n_examples = 0
        self.seen = []          # class indices (config order) with at least one example, ascending
        self.counts = {}        # class name -> number of examples

    # -- subclasses implement these -------------------------------------------------------------
    def _fit(self, x, y):
        raise NotImplementedError

    def _seen_proba(self, x):
        """(N, len(seen)) distribution over the seen classes."""
        raise NotImplementedError

    def _usable(self):
        return len(self.seen) >= 1

    def _state(self):
        raise NotImplementedError

    def _set_state(self, arrays, params):
        raise NotImplementedError

    # -- shared ---------------------------------------------------------------------------------
    def fit(self, features, labels):
        x = np.asarray(features, dtype=np.float64)
        unknown = set(labels) - set(self.classes)
        if unknown:
            raise ValueError(f"labels not in the config class order: {sorted(unknown)}")
        y = np.array([self.classes.index(l) for l in labels], dtype=int)
        self.n_examples = len(y)
        self.seen = sorted(set(y.tolist()))
        self.counts = {self.classes[c]: int((y == c).sum()) for c in self.seen}
        if self.n_examples:
            self._fit(x, y)
        return self

    def predict_proba(self, features, generic_probs):
        generic = np.asarray(generic_probs, dtype=np.float64)
        if not self._usable():
            return generic.copy()
        x = np.asarray(features, dtype=np.float64)
        seen = np.array(self.seen)
        out = generic.copy()
        out[:, seen] = self._seen_proba(x) * generic[:, seen].sum(axis=1, keepdims=True)
        return out

    # -- persistence ----------------------------------------------------------------------------
    def save(self, user_id, model_name, model_sha256, root=USERS_DIR):
        folder = Path(root) / user_id
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f"{model_name}_{self.kind}"
        np.savez(stem.with_suffix(".npz"), **self._state())
        meta = {"kind": self.kind, "user_id": user_id, "model": model_name, "model_sha256": model_sha256,
                "class_order": self.classes, "n_examples": self.n_examples, "counts": self.counts,
                "seen": self.seen, "params": self._params()}
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return stem

    def _params(self):
        return {}


class LogisticAdapter(Adapter):
    kind = "logistic"

    def __init__(self, classes, C=0.1, max_iter=1000):
        super().__init__(classes)
        self.C, self.max_iter = C, max_iter
        self.coef = self.intercept = None

    def _usable(self):
        return self.coef is not None and len(self.seen) >= 2

    def _fit(self, x, y):
        if len(self.seen) < 2:
            self.coef = self.intercept = None
            return
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=self.max_iter).fit(x, y)
        assert list(clf.classes_) == self.seen
        self.coef, self.intercept = clf.coef_, clf.intercept_

    def _seen_proba(self, x):
        z = x @ self.coef.T + self.intercept
        if self.coef.shape[0] == 1:  # two classes: sklearn stores one row
            p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
            return np.stack([1.0 - p1, p1], axis=1)
        return _softmax(z)

    def _state(self):
        return {"coef": self.coef if self.coef is not None else np.zeros((0, 0)),
                "intercept": self.intercept if self.intercept is not None else np.zeros(0)}

    def _params(self):
        return {"C": self.C, "max_iter": self.max_iter}


class PrototypeAdapter(Adapter):
    kind = "prototype"

    def __init__(self, classes, temperature=0.1):
        super().__init__(classes)
        self.temperature = temperature
        self.prototypes = None  # (len(seen), dim), L2-normalised class means

    def _fit(self, x, y):
        self.prototypes = _l2_normalise(np.stack([x[y == c].mean(axis=0) for c in self.seen]))

    def _seen_proba(self, x):
        return _softmax(_l2_normalise(x) @ self.prototypes.T / self.temperature)

    def _state(self):
        return {"prototypes": self.prototypes if self.prototypes is not None else np.zeros((0, 0))}

    def _params(self):
        return {"temperature": self.temperature}


KINDS = {"logistic": LogisticAdapter, "prototype": PrototypeAdapter}


def make_adapter(kind, classes, cfg):
    """Build an unfitted adapter from the `personalize` block of configs/base.yaml."""
    if kind == "logistic":
        return LogisticAdapter(classes, C=cfg["logistic"]["C"])
    if kind == "prototype":
        return PrototypeAdapter(classes, temperature=cfg["prototype"]["temperature"])
    raise ValueError(f"unknown adapter {kind!r}; choose from {list(KINDS)}")


def load_adapter(user_id, model_name, kind, expected_sha256, expected_classes, root=USERS_DIR):
    """Load a saved adapter. Refuses one made with a different base model file or class order."""
    stem = Path(root) / user_id / f"{model_name}_{kind}"
    meta = json.loads(stem.with_suffix(".json").read_text())
    if meta["model_sha256"] != expected_sha256:
        raise ValueError(f"{stem.name}: saved for a different base model file (sha256 differs)")
    if meta["class_order"] != list(expected_classes):
        raise ValueError(f"{stem.name}: saved class order {meta['class_order']} != {list(expected_classes)}")
    arrays = np.load(stem.with_suffix(".npz"))
    a = KINDS[kind](meta["class_order"], **meta["params"])
    a.n_examples, a.seen, a.counts = meta["n_examples"], meta["seen"], meta["counts"]
    if kind == "logistic":
        if arrays["coef"].size:
            a.coef, a.intercept = arrays["coef"], arrays["intercept"]
    else:
        if arrays["prototypes"].size:
            a.prototypes = arrays["prototypes"]
    return a
