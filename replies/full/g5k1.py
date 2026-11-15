import numpy as np
from scipy import ndimage
from llm_mrg import blocks as B

DESCRIPTION = ("Multichannel region growing on the first four PCs with a running-mean criterion and "
               "seeds ordered from the most homogeneous pixel outward; regions capped by a 3x3 grid; "
               "pixel vote of an RBF SVM on PCs, smoothed PCs and local PC standard deviation.")
PARAMS = {"n_pc": 4, "threshold": 0.9, "grad_sigma": 1.0, "grid": 3,
          "ctx_sigma1": 1.0, "ctx_sigma2": 3.0, "std_size": 5}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    f = np.asarray(ctx.pcs[..., : params["n_pc"]], dtype=np.float64)
    g = sum(ndimage.gaussian_gradient_magnitude(f[..., i], params["grad_sigma"]) for i in range(f.shape[2]))
    order = np.argsort(g.ravel(), kind="stable")
    cells = B.grid_cells(ctx.H, ctx.W, params["grid"])
    return B.grow(ctx, f, order=order, threshold=params["threshold"], criterion="mean", cells=cells)


def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    k = p.shape[2]
    s1 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma1"]) for i in range(k)], -1)
    s2 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma2"]) for i in range(k)], -1)
    sz = params["std_size"]
    mu = np.stack([ndimage.uniform_filter(p[..., i], sz) for i in range(k)], -1)
    sq = np.stack([ndimage.uniform_filter(p[..., i] ** 2, sz) for i in range(k)], -1)
    return np.concatenate([p, s1, s2, np.sqrt(np.maximum(sq - mu ** 2, 0))], -1)
