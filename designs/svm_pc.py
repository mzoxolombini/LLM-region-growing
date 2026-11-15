"""Control: plain pixel-wise RBF SVM on the ten standardised PCs. No smoothing, no regions."""
import numpy as np

DESCRIPTION = "Pixel-wise RBF SVM on the ten PCs; no spatial information."
PARAMS = {}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    return np.arange(ctx.H * ctx.W, dtype=np.int64).reshape(ctx.H, ctx.W)


def pixel_features(ctx, params):
    return np.asarray(ctx.pcs, dtype=np.float32)
