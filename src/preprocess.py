"""Face detection and preprocessing for the frozen base model.

Matches what the base model saw in training (Keras flow_from_directory in the old repo):
    PIL grayscale conversion ("L")  ->  float32  ->  x * (1/255)
The old repo's webcam demo used OpenCV's grayscale instead, which differs from PIL's by one grey
level in about 9% of RAF-DB images; here PIL's conversion is used for everything.

Training images were already cropped, face-sized (RAF-DB 100x100, FER2013 48x48), so training
never resized. A live crop is shrunk with cv2.INTER_AREA. Image files are decoded with PIL, as in
training, because OpenCV's JPEG decoder can give slightly different pixels.
"""

import cv2
import numpy as np
from PIL import Image

INTERPOLATION = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}


def load_face_detector(cfg):
    """OpenCV Haar cascade, as in the old repo's webcam demo."""
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + cfg["cascade"])
    assert not detector.empty(), f"could not load the Haar cascade {cfg['cascade']}"
    return detector


def find_largest_face(detector, frame_bgr, cfg):
    """(x, y, w, h) of the biggest face in a BGR frame, or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)  # detection only; the model input uses PIL
    faces = detector.detectMultiScale(gray, scaleFactor=cfg["scale_factor"], minNeighbors=cfg["min_neighbors"],
                                      minSize=(cfg["min_size"], cfg["min_size"]))
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def crop(frame, box):
    x, y, w, h = box
    return frame[y:y + h, x:x + w]


def to_gray(face, bgr=True):
    """uint8 grayscale (H, W) using PIL's "L" conversion, the one training used.

    face: an (H, W) gray array, or an (H, W, 3) array in BGR (OpenCV) or RGB order.
    """
    if face.ndim == 2:
        return face
    rgb = face[:, :, ::-1] if bgr else face
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).convert("L"))


def load_gray_file(path):
    """Decode an image file exactly as Keras' load_img(color_mode='grayscale') did in training."""
    with Image.open(path) as img:
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")
        return np.asarray(img)


def to_model_input(gray, input_size, cfg):
    """uint8 (H, W) gray face -> float32 (1, H, W, 1) model input."""
    h, w = input_size
    if gray.shape != (h, w):
        gray = cv2.resize(gray, (w, h), interpolation=INTERPOLATION[cfg["resize_interpolation"]])
    x = gray.astype("float32")[None, :, :, None]
    x *= cfg["rescale"]  # in-place float32 multiply, bit-identical to ImageDataGenerator.standardize
    return x
