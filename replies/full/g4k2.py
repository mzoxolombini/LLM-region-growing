"""E2 of g03_01 (SLIC regions: best OA_val/mF1 on Indian Pines, 99.2 / 0.976) and g03_02 (MRG and
watershed intersection with pixel vote: best OA_grp on Indian Pines, 96.0). Regions are the
intersection of MRG regions and SLIC superpixels, labelled by g03_02's pixel vote."""
import numpy as np
from scipy import ndimage
from skimage.segmentation import slic
from llm_mrg import blocks as B

DESCRIPTION = ("Intersection of MRG regions and 12-pixel SLIC superpixels on six PCs, labelled by "
               "pixel vote of an RBF SVM on PCs and smoothed PCs.")
PARAMS = {"t_intensity": 0.05, "t_orientation": 0.7853981633974483, "n_bins": 256, "grid": 3,
          "connectivity": 8, "n_pc": 6, "pixels_per_segment": 12, "compactness": 0.1, "sigma": 0.5,
          "ctx_sigma": 2.0}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    a = B.mrg(ctx, t_intensity=params["t_intensity"], t_orientation=params["t_orientation"],
              n_bins=params["n_bins"], grid=params["grid"], connectivity=params["connectivity"])
    f = np.asarray(ctx.pcs[..., : params["n_pc"]], dtype=np.float64)
    n = max(16, int(ctx.H * ctx.W / params["pixels_per_segment"]))
    b = slic(f, n_segments=n, compactness=params["compactness"], sigma=params["sigma"],
             channel_axis=-1, start_label=0, enforce_connectivity=True).astype(np.int64)
    return B.relabel(a.astype(np.int64) * (int(b.max()) + 1) + b)


def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    s = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma"]) for i in range(p.shape[2])], -1)
    return np.concatenate([p, s], -1)
