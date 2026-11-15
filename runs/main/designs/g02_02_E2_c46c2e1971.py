"""E2: combines the superpixel regions of g01_01 (best OA_grp on four scenes) with the PC
statistics of g01_02 (best OA_grp on Indian Pines, where 26-pixel superpixels were too coarse).
Smaller superpixels for the small fields, PC mean/std + V + mean spectrum as in g01_02, plus a
context term: the mean of 2-pixel-smoothed PCs over the region."""
import numpy as np
from scipy import ndimage
from skimage.segmentation import slic
from llm_mrg import blocks as B

DESCRIPTION = ("SLIC superpixels of about 15 pixels on five PCs, described by the mean and standard "
               "deviation of the ten PCs, mean V, mean spectrum and mean smoothed-PC context, "
               "labelled by an RBF SVM.")

PARAMS = {"n_pc": 5, "pixels_per_segment": 15, "compactness": 0.1, "sigma": 0.5, "max_iter": 10,
          "ctx_sigma": 2.0}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "region"


def segment(ctx, params):
    f = np.asarray(ctx.pcs[..., : params["n_pc"]], dtype=np.float64)
    n_seg = max(16, int(ctx.H * ctx.W / params["pixels_per_segment"]))
    lab = slic(f, n_segments=n_seg, compactness=params["compactness"], sigma=params["sigma"],
               max_num_iter=params["max_iter"], channel_axis=-1, start_label=0,
               enforce_connectivity=True)
    return B.relabel(lab)


def region_features(ctx, regions, rows, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    sm = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma"]) for i in range(p.shape[2])], -1)
    return np.hstack([B.region_means(p, regions, rows), B.region_stds(p, regions, rows),
                      B.region_means(ctx.v, regions, rows), B.region_means(ctx.bands, regions, rows),
                      B.region_means(sm, regions, rows)])
