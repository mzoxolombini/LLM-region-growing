import numpy as np
from scipy import ndimage
from skimage.segmentation import slic
from llm_mrg import blocks as B

DESCRIPTION = ("SLIC superpixels of about 12 pixels on the first six PCs, each described by PC "
               "mean/std, mean spectrum and the mean of 2-pixel smoothed PCs, RBF SVM on regions.")
PARAMS = {"n_pc": 6, "pixels_per_segment": 12, "compactness": 0.1, "sigma": 0.5, "ctx_sigma": 2.0}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "region"


def segment(ctx, params):
    f = np.asarray(ctx.pcs[..., : params["n_pc"]], dtype=np.float64)
    n = max(16, int(ctx.H * ctx.W / params["pixels_per_segment"]))
    lab = slic(f, n_segments=n, compactness=params["compactness"], sigma=params["sigma"],
               channel_axis=-1, start_label=0, enforce_connectivity=True)
    return B.relabel(lab)


def region_features(ctx, regions, rows, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    s = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma"]) for i in range(p.shape[2])], -1)
    return np.hstack([B.region_means(p, regions, rows), B.region_stds(p, regions, rows),
                      B.region_means(ctx.bands, regions, rows), B.region_means(s, regions, rows)])
