"""Freezing, class order and preprocessing checks for the base model.  Run: pytest"""

import numpy as np
import pytest

from src import preprocess
from src.base_model import BaseModel, load_config

MODELS = list(load_config()["models"])


@pytest.fixture(scope="module", params=MODELS)
def base(request):
    return BaseModel(request.param)


def snapshot(model):
    """Every weight, including the BatchNorm moving mean/variance (non-trainable weights)."""
    return [w.numpy().copy() for w in model.weights]


def random_input(base, n=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, *base.input_size, 1), dtype=np.float32)


def test_all_layers_frozen(base):
    assert not base.model.trainable
    assert all(not layer.trainable for layer in base.model.layers)
    assert base.model.trainable_weights == []
    assert getattr(base.model, "optimizer", None) is None  # loaded uncompiled: nothing can train it


def test_forward_pass_changes_no_weights(base):
    before = snapshot(base.model)
    x = random_input(base)
    base.predict_input(x)
    # training=True is what would update BatchNorm statistics in an unfrozen model.
    base.model(x, training=True)
    base.model.predict(x, verbose=0)
    after = snapshot(base.model)
    assert len(before) == len(after) > 0
    for a, b in zip(before, after):
        assert np.array_equal(a, b)


def test_probabilities(base):
    p = base.predict_input(random_input(base))
    assert p.shape == (4, 7)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_class_reordering(base):
    x = random_input(base, n=2)
    native = base.native_probs(x)
    reordered = base.predict_input(x)
    for i, name in enumerate(base.classes):
        assert np.array_equal(reordered[:, i], native[:, base.native_classes.index(name)])


def test_same_output_eager_and_graph(base):
    """Guards against the tensorflow-metal fused Dense kernel bug (see load_frozen)."""
    import tensorflow as tf

    x = random_input(base)
    eager = base.model(x, training=False).numpy()
    graph = tf.function(lambda t: base.model(t, training=False))(x).numpy()
    assert np.allclose(eager, graph, atol=1e-5)
    assert base.model.get_layer("dense").activation.__name__ == "linear"


def test_input_shape_matches_model(base):
    assert base.model.input_shape == (None, *base.input_size, 1)


def test_preprocess_range_and_shape(base):
    gray = np.full((300, 240), 255, dtype=np.uint8)
    x = preprocess.to_model_input(gray, base.input_size, base.prep)
    assert x.shape == (1, *base.input_size, 1) and x.dtype == np.float32
    assert np.isclose(x.max(), 1.0) and x.min() >= 0.0


def test_gray_matches_pil():
    """BGR crops are converted with PIL's formula, the one used in training (not OpenCV's)."""
    from PIL import Image

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    expected = np.asarray(Image.fromarray(rgb).convert("L"))
    assert np.array_equal(preprocess.to_gray(rgb[:, :, ::-1], bgr=True), expected)
