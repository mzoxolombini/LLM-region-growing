"""M1 on g01_02. Its regions are kept (MRG growth rule untouched); feedback shows OA_grp of
about 84 on Indian Pines and Houston, so held-out regions are still confused. Regions of 3-5
pixels carry little information on their own, so each region is also described by its
spatial context: PCs smoothed at two scales and averaged over the region. RBF kernel because
the class boundaries in PC space are not linear."""
import numpy as np
from scipy import ndimage
from llm_mrg import blocks as B

DESCRIPTION = ("Baseline MRG growth rule unchanged; regions described by PC mean and std, mean V, "
               "mean spectrum and two-scale smoothed-PC context, labelled by an RBF SVM.")

PARAMS = {
    "t_intensity": 0.05,
    "t_orientation": 0.7853981633974483,
    "n_bins": 256,
    "grid": 3,
    "connectivity": 8,
    "ctx_sigma1": 1.5,
    "ctx_sigma2": 4.0,
}

CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "region"


def segment(ctx, params):
    return B.mrg(ctx, t_intensity=params["t_intensity"], t_orientation=params["t_orientation"],
                 n_bins=params["n_bins"], grid=params["grid"],
                 connectivity=params["connectivity"])


def region_features(ctx, regions, rows, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    s1 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma1"]) for i in range(p.shape[2])], -1)
    s2 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma2"]) for i in range(p.shape[2])], -1)
    return np.hstack([B.region_means(p, regions, rows), B.region_stds(p, regions, rows),
                      B.region_means(ctx.v, regions, rows), B.region_means(ctx.bands, regions, rows),
                      B.region_means(s1, regions, rows), B.region_means(s2, regions, rows)])
