"""Competitor: the pixel classifier of LLM-MRG (g01_03) with no regions. Every pixel is its own
region, so the 'vote' is the pixel's own prediction."""
import numpy as np
from scipy import ndimage

DESCRIPTION = "Pixel-wise RBF SVM on PCs and 2-pixel smoothed PCs (LLM-MRG features, no regions)."
PARAMS = {"ctx_sigma": 2.0}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    return np.arange(ctx.H * ctx.W, dtype=np.int64).reshape(ctx.H, ctx.W)


def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    s = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma"]) for i in range(p.shape[2])], -1)
    return np.concatenate([p, s], -1)
