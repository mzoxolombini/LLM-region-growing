"""M1 on g04_01. Weakest point: Indian Pines OA_grp 95.3. Gaussian context mixes neighbouring
fields at their borders; replace part of it by object context from a coarser segmentation."""
import numpy as np
from scipy import ndimage
from skimage.segmentation import felzenszwalb
from llm_mrg import blocks as B

DESCRIPTION = ("Felzenszwalb segmentation on three PCs labelled by pixel vote of an RBF SVM whose "
               "pixel description adds object context: the mean PCs of the pixel's segment in a "
               "coarser Felzenszwalb partition, which follows field borders unlike Gaussian smoothing.")
PARAMS = {"n_pc_seg": 3, "scale": 10.0, "sigma": 0.5, "min_size": 5,
          "ctx_sigma1": 1.0, "ctx_sigma2": 3.0, "std_size": 5,
          "coarse_scale": 60.0, "coarse_min_size": 20}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    f = np.asarray(ctx.pcs[..., : params["n_pc_seg"]], dtype=np.float64)
    return B.relabel(felzenszwalb(f, scale=params["scale"], sigma=params["sigma"],
                                  min_size=params["min_size"], channel_axis=-1))


def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    k = p.shape[2]
    s1 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma1"]) for i in range(k)], -1)
    s2 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma2"]) for i in range(k)], -1)
    sz = params["std_size"]
    mu = np.stack([ndimage.uniform_filter(p[..., i], sz) for i in range(k)], -1)
    sq = np.stack([ndimage.uniform_filter(p[..., i] ** 2, sz) for i in range(k)], -1)
    f = np.asarray(ctx.pcs[..., : params["n_pc_seg"]], dtype=np.float64)
    coarse = B.relabel(felzenszwalb(f, scale=params["coarse_scale"], sigma=params["sigma"],
                                    min_size=params["coarse_min_size"], channel_axis=-1))
    obj = B.region_means(p, coarse)[coarse].astype(np.float32)
    return np.concatenate([p, s1, s2, np.sqrt(np.maximum(sq - mu ** 2, 0)), obj], -1)
