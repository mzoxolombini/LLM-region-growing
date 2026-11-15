"""Competitor: the pixel classifier of LLM-RB (g10_04) with no regions."""
import numpy as np
from scipy import ndimage

DESCRIPTION = "Pixel-wise RBF SVM on the LLM-RB pixel features, no regions."
PARAMS = {"n_pc_seg": 3, "scale": 10.0, "sigma": 0.5, "min_size": 5, "n_pc": 5, "grad_sigma": 1.0,
          "marker_step": 2, "compactness": 0.01, "ctx_sigma1": 1.0, "ctx_sigma2": 5.0, "std_size": 11}
CLASSIFIER = {"kernel": "rbf", "C": 3000.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    return np.arange(ctx.H * ctx.W, dtype=np.int64).reshape(ctx.H, ctx.W)


def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    k = p.shape[2]
    s1 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma1"]) for i in range(k)], -1)
    s2 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma2"]) for i in range(k)], -1)
    sz = params["std_size"]
    mu = np.stack([ndimage.uniform_filter(p[..., i], sz) for i in range(k)], -1)
    sq = np.stack([ndimage.uniform_filter(p[..., i] ** 2, sz) for i in range(k)], -1)
    return np.concatenate([p, s1, s2, np.sqrt(np.maximum(sq - mu ** 2, 0))], -1)
