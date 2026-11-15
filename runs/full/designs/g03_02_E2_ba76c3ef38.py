"""E2 of g02_04 (MRG regions, best IP: F 94.8) and g02_02 (watershed, best elsewhere). Regions are
the intersection of the two partitions, so a region must be homogeneous under both criteria;
labelling is the shared pixel vote."""
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from llm_mrg import blocks as B

DESCRIPTION = ("Intersection of MRG regions and a 3-pixel-lattice PC-gradient watershed, labelled by "
               "pixel vote of an RBF SVM on PCs and smoothed PCs.")
PARAMS = {"t_intensity": 0.05, "t_orientation": 0.7853981633974483, "n_bins": 256, "grid": 3,
          "connectivity": 8, "n_pc": 5, "grad_sigma": 1.0, "marker_step": 3, "compactness": 0.01,
          "ctx_sigma": 2.0}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    a = B.mrg(ctx, t_intensity=params["t_intensity"], t_orientation=params["t_orientation"],
              n_bins=params["n_bins"], grid=params["grid"], connectivity=params["connectivity"])
    p = np.asarray(ctx.pcs[..., : params["n_pc"]], dtype=np.float64)
    g = sum(ndimage.gaussian_gradient_magnitude(p[..., i], params["grad_sigma"]) for i in range(p.shape[2]))
    st = params["marker_step"]
    m = np.zeros(g.shape, np.int32)
    yy, xx = np.mgrid[st // 2: g.shape[0]: st, st // 2: g.shape[1]: st]
    m[yy, xx] = np.arange(1, yy.size + 1).reshape(yy.shape)
    b = watershed(g, m, compactness=params["compactness"]).astype(np.int64)
    return B.relabel(a.astype(np.int64) * (int(b.max()) + 1) + b)


def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    s = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma"]) for i in range(p.shape[2])], -1)
    return np.concatenate([p, s], -1)
