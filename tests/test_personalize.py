"""Adapter, blending and persistence checks with small synthetic data.  Run: pytest"""

import numpy as np
import pytest

from src.base_model import load_config
from src.personalize import LogisticAdapter, PrototypeAdapter, blend, load_adapter, make_adapter

CFG = load_config()
CLASSES = CFG["classes"]
DIM = 16
ADAPTERS = [LogisticAdapter, PrototypeAdapter]


def make_data(labels, per_class=6, seed=0, noise=0.3):
    """Each class is a distinct direction in feature space plus noise."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(CLASSES), DIM))
    xs, ys = [], []
    for name in labels:
        c = CLASSES.index(name)
        xs.append(centers[c] + noise * rng.normal(size=(per_class, DIM)))
        ys += [name] * per_class
    return np.vstack(xs), ys, centers


def uniform(n):
    return np.full((n, 7), 1 / 7)


def is_distribution(p):
    return np.all(p >= 0) and np.allclose(p.sum(axis=1), 1.0)


@pytest.mark.parametrize("cls", ADAPTERS)
def test_learns_seen_classes(cls):
    x, y, _ = make_data(CLASSES)
    xt, yt, _ = make_data(CLASSES, seed=0, per_class=4, noise=0.3)
    p = cls(CLASSES).fit(x, y).predict_proba(xt, uniform(len(xt)))
    assert is_distribution(p)
    assert (np.array(CLASSES)[p.argmax(1)] == np.array(yt)).mean() > 0.9


@pytest.mark.parametrize("cls", ADAPTERS)
def test_missing_classes_keep_generic_probability(cls):
    seen = ["Happy", "Sad", "Angry"]
    x, y, centers = make_data(seen)
    rng = np.random.default_rng(1)
    generic = rng.dirichlet(np.ones(7), size=5)
    p = cls(CLASSES).fit(x, y).predict_proba(rng.normal(size=(5, DIM)), generic)
    assert is_distribution(p)
    unseen = [i for i, c in enumerate(CLASSES) if c not in seen]
    assert np.allclose(p[:, unseen], generic[:, unseen])           # unseen classes untouched
    s = [CLASSES.index(c) for c in seen]
    assert np.allclose(p[:, s].sum(1), generic[:, s].sum(1))       # seen mass preserved


@pytest.mark.parametrize("cls", ADAPTERS)
def test_single_class_and_no_examples_return_generic(cls):
    generic = np.random.default_rng(2).dirichlet(np.ones(7), size=4)
    xt = np.random.default_rng(3).normal(size=(4, DIM))
    one = cls(CLASSES).fit(*make_data(["Happy"])[:2])
    assert np.allclose(one.predict_proba(xt, generic), generic)
    none = cls(CLASSES).fit(np.zeros((0, DIM)), [])
    assert none.n_examples == 0 and np.allclose(none.predict_proba(xt, generic), generic)


def test_two_class_logistic_uses_binary_path():
    x, y, _ = make_data(["Happy", "Sad"])
    p = LogisticAdapter(CLASSES).fit(x, y).predict_proba(x, uniform(len(x)))
    assert is_distribution(p) and (np.array(CLASSES)[p.argmax(1)] == np.array(y)).all()


def test_logistic_balances_classes():
    rng = np.random.default_rng(4)
    x = np.vstack([rng.normal(1, 1, (50, DIM)), rng.normal(-1, 1, (2, DIM))])
    y = ["Happy"] * 50 + ["Sad"] * 2
    p = LogisticAdapter(CLASSES, C=1.0).fit(x, y)._seen_proba(np.full((1, DIM), -1.0))
    assert p[0, 1] > 0.5  # the rare class is not swamped by the 50 Happy examples


def test_prototypes_are_unit_length_means():
    x, y, _ = make_data(["Happy", "Sad"])
    a = PrototypeAdapter(CLASSES).fit(x, y)
    assert np.allclose(np.linalg.norm(a.prototypes, axis=1), 1.0)
    h = x[np.array(y) == "Happy"].mean(0)
    assert np.allclose(a.prototypes[0], h / np.linalg.norm(h))


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        LogisticAdapter(CLASSES).fit(np.zeros((2, DIM)), ["Happy", "Joy"])


def test_blend_weights():
    pu, pg = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert np.allclose(blend(pu, pg, n=0, k0=8), pg)                 # no examples: generic
    assert np.allclose(blend(pu, pg, n=8, k0=8), [[0.5, 0.5]])       # n = k0: half and half
    assert np.allclose(blend(pu, pg, n=24, k0=8), [[0.75, 0.25]])
    p = blend(np.full((3, 7), 1 / 7), np.random.default_rng(5).dirichlet(np.ones(7), 3), 5, 8)
    assert is_distribution(p)


@pytest.mark.parametrize("kind", ["logistic", "prototype"])
def test_save_load_round_trip(tmp_path, kind):
    x, y, _ = make_data(["Happy", "Sad", "Fear"])
    a = make_adapter(kind, CLASSES, CFG["personalize"]).fit(x, y)
    a.save("P01", "rafdb_clean", "sha-a", root=tmp_path)
    b = load_adapter("P01", "rafdb_clean", kind, "sha-a", CLASSES, root=tmp_path)
    generic = np.random.default_rng(6).dirichlet(np.ones(7), size=5)
    xt = np.random.default_rng(7).normal(size=(5, DIM))
    assert np.allclose(a.predict_proba(xt, generic), b.predict_proba(xt, generic))
    assert b.n_examples == 18 and b.counts == {"Fear": 6, "Happy": 6, "Sad": 6}


def test_load_refuses_wrong_model_or_class_order(tmp_path):
    x, y, _ = make_data(["Happy", "Sad"])
    LogisticAdapter(CLASSES).fit(x, y).save("P01", "rafdb_clean", "sha-a", root=tmp_path)
    with pytest.raises(ValueError, match="different base model"):
        load_adapter("P01", "rafdb_clean", "logistic", "sha-b", CLASSES, root=tmp_path)
    with pytest.raises(ValueError, match="class order"):
        load_adapter("P01", "rafdb_clean", "logistic", "sha-a", CLASSES[::-1], root=tmp_path)


def test_users_folder_is_gitignored():
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    assert subprocess.run(["git", "check-ignore", "-q", "models/users/P01/rafdb_clean_logistic.npz"],
                          cwd=root).returncode == 0
