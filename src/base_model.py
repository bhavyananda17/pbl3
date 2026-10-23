"""The frozen base model: load it, preprocess a face, return 7 emotion probabilities.

    from src.base_model import BaseModel
    base = BaseModel()                        # RAF-DB clean model (configs/base.yaml default)
    base = BaseModel("fer2013_clean")         # or the FER2013 one
    probs = base.predict_frame(frame_bgr)     # webcam frame -> probabilities, or None if no face
    probs = base.predict_file("face.jpg")     # already-cropped face image

Probabilities are always in the order of `classes` in configs/base.yaml, whichever model is loaded.
Every layer is set non-trainable, and the model is loaded without an optimizer, so nothing in this
project can change its weights. Per-user learning will sit on top of it (models/users/).
"""

import hashlib
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's startup chatter

import numpy as np
import yaml

from src import preprocess

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "base.yaml"


def load_config(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen(path):
    """Load a .keras model with no optimizer and every layer non-trainable.

    The base models were trained with tensorflow-metal, whose fused MatMul+BiasAdd+ReLU kernel
    (used in graph mode, i.e. fit/predict) silently drops the ReLU. So the network the weights
    were trained as has a LINEAR first Dense layer, even though its saved config says 'relu'.
    Run as saved, results depend on the device: 83.2% on RAF-DB with GPU model.predict, 67.8%
    on CPU or eager. Setting the activation to linear here gives the trained-as network on every
    device and mode (83.2%, same class as the old repo's predictions on all test images).
    """
    import keras

    model = keras.models.load_model(path, compile=False)
    dense = model.get_layer("dense")
    assert dense.activation is keras.activations.relu, "unexpected base model: first Dense is not relu"
    dense.activation = keras.activations.linear
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False
    return model


class BaseModel:
    def __init__(self, name=None, config_path=CONFIG, check_hash=True):
        self.cfg = load_config(config_path)
        self.name = name or self.cfg["default_model"]
        if self.name not in self.cfg["models"]:
            raise ValueError(f"unknown model {self.name!r}; choose from {list(self.cfg['models'])}")
        m = self.cfg["models"][self.name]
        path = ROOT / m["path"]
        if not path.exists():
            raise SystemExit(f"No model file at {m['path']}. See README (Setup).")
        if check_hash and sha256(path) != m["sha256"]:
            raise SystemExit(f"{m['path']} does not match the sha256 in {Path(config_path).name}.")

        self.model = load_frozen(path)
        self.input_size = tuple(m["input_size"])
        self.classes = list(self.cfg["classes"])
        self.native_classes = list(m["native_classes"])
        assert sorted(self.native_classes) == sorted(self.classes), f"{self.name}: class names differ"
        # order[i] = index in the model's output of self.classes[i]
        self.order = np.array([self.native_classes.index(c) for c in self.classes])
        self.prep = self.cfg["preprocessing"]
        self._detector = None

        import keras

        self.feature_layer = self.cfg["features"]["layer"]
        self.feature_dim = self.cfg["features"]["dim"]
        # Re-chains the same frozen layers (shared, not copied) to expose the feature layer.
        inp = keras.Input(shape=(*self.input_size, 1))
        x, feats = inp, None
        for layer in self.model.layers:
            x = layer(x)
            if layer.name == self.feature_layer:
                feats = x
        assert feats is not None, f"{self.name}: no layer named {self.feature_layer}"
        self._with_features = keras.Model(inp, [feats, x])
        assert self._with_features.outputs[0].shape[-1] == self.feature_dim

    # Model-native output, before reordering. Used by the match check against the old repo.
    def native_probs(self, x):
        return self.model(x, training=False).numpy()

    def predict_input(self, x):
        """(N, H, W, 1) float32 model input -> (N, 7) probabilities in `classes` order."""
        return self.native_probs(x)[:, self.order]

    def predict_with_features(self, x):
        """(N, H, W, 1) float32 model input -> ((N, 7) probabilities in `classes` order, (N, 512) features).

        Features are the output of `features.layer` in configs/base.yaml (batch_normalization_8),
        the vector the final softmax layer sees.
        """
        feats, probs = self._with_features(x, training=False)
        return probs.numpy()[:, self.order], feats.numpy()

    def predict_gray(self, gray):
        """uint8 (H, W) gray face crop -> 7 probabilities."""
        return self.predict_input(preprocess.to_model_input(gray, self.input_size, self.prep))[0]

    def predict_face(self, face_bgr):
        """BGR face crop (e.g. from OpenCV) -> 7 probabilities."""
        return self.predict_gray(preprocess.to_gray(face_bgr, bgr=True))

    def predict_file(self, path):
        """Image file holding an already-cropped face -> 7 probabilities."""
        return self.predict_gray(preprocess.load_gray_file(path))

    def predict_frame(self, frame_bgr):
        """Full BGR frame -> (probabilities, face box), or (None, None) when no face is found."""
        if self._detector is None:
            self._detector = preprocess.load_face_detector(self.prep["face_detector"])
        box = preprocess.find_largest_face(self._detector, frame_bgr, self.prep["face_detector"])
        if box is None:
            return None, None
        return self.predict_face(preprocess.crop(frame_bgr, box)), box

    def predict_face_with_features(self, face_bgr):
        """BGR face crop -> (7 probabilities, 512 features)."""
        x = preprocess.to_model_input(preprocess.to_gray(face_bgr, bgr=True), self.input_size, self.prep)
        probs, feats = self.predict_with_features(x)
        return probs[0], feats[0]
