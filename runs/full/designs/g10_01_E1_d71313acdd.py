"""E1. Every design in the population forms regions from Felzenszwalb x watershed partitions on PCs.
This design returns to the original MRG growth rule (histogram-frequency seeds, intensity and
orientation constraints to the seed, 3x3 grid) for region formation, and pairs it with the pixel
description that the population converged on."""
import numpy as np
from scipy import ndimage
from llm_mrg import blocks as B

DESCRIPTION = ("Original MRG growth rule on the V channel for regions; pixel vote of an RBF SVM on "
               "PCs, two smoothing scales, local PC std and 30 smoothed raw bands.")
PARAMS = {"t_intensity": 0.05, "t_orientation": 0.7853981633974483, "n_bins": 256, "grid": 3,
          "connectivity": 8, ctx_sigma1": 1.0, "ctx_sigma2": 5.0, "std_size": 9,
          "n_bands": 30, "band_sigma": 1.0}
CLASSIFIER = {"kernel": "rbf", "C": 1000.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    return B.mrg(ctx, t_intensity=params["t_intensity"], t_orientation=params["t_orientation"],
                 n_bins=params["n_bins"], grid=params["grid"], connectivity=params["connectivity"])


def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    k = p.shape[2]
    s1 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma1"]) for i in range(k)], -1)
    s2 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma2"]) for i in range(k)], -1)
    sz = params["std_size"]
    mu = np.stack([ndimage.uniform_filter(p[..., i], sz) for i in range(k)], -1)
    sq = np.stack([ndimage.uniform_filter(p[..., i] ** 2, sz) for i in range(k)], -1)
    bsel = np.linspace(0, ctx.B - 1, params["n_bands"]).astype(int)
    bands = np.stack([ndimage.gaussian_filter(np.asarray(ctx.bands[..., j], np.float32), params["band_sigma"])
                      for j in bsel], -1)
    return np.concatenate([p, s1, s2, np.sqrt(np.maximum(sq - mu ** 2, 0)), bands], -1)
