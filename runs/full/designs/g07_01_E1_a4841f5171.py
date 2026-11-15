import numpy as np
from scipy import ndimage
from skimage.segmentation import slic
from skimage.morphology import disk, opening, closing
from llm_mrg import blocks as B

DESCRIPTION = ("Very fine SLIC superpixels (about 5 pixels) on six PCs labelled by pixel vote of an "
               "RBF SVM on PCs, smoothed PCs, local PC std and an extended morphological profile.")
PARAMS = {"n_pc": 6, "pixels_per_segment": 5, "compactness": 0.1, "sigma": 0.5,
          "ctx_sigma1": 1.0, "ctx_sigma2": 3.0, "std_size": 5, "emp_pcs": 3, "emp_radii": [1, 3, 5]}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    f = np.asarray(ctx.pcs[..., : params["n_pc"]], dtype=np.float64)
    n = max(16, int(ctx.H * ctx.W / params["pixels_per_segment"]))
    return B.relabel(slic(f, n_segments=n, compactness=params["compactness"], sigma=params["sigma"],
                          channel_axis=-1, start_label=0, enforce_connectivity=True))


def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    k = p.shape[2]
    s1 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma1"]) for i in range(k)], -1)
    s2 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma2"]) for i in range(k)], -1)
    sz = params["std_size"]
    mu = np.stack([ndimage.uniform_filter(p[..., i], sz) for i in range(k)], -1)
    sq = np.stack([ndimage.uniform_filter(p[..., i] ** 2, sz) for i in range(k)], -1)
    sd = np.sqrt(np.maximum(sq - mu ** 2, 0))
    return np.concatenate([p, s1, s2, sd], -1)
