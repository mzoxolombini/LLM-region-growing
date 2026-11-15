"""E2 of g06_02 (Felzenszwalb x watershed regions, tuned context, F 99.74) and g06_03 (extended
morphological profile in the pixel description, F 99.60). Regions and settings of g06_02 with
the EMP features of g06_03."""
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed, felzenszwalb
from skimage.morphology import disk, opening, closing
from llm_mrg import blocks as B

DESCRIPTION = ("Felzenszwalb x 2-pixel-lattice watershed intersection; pixel vote of an RBF SVM on "
               "PCs, two smoothing scales, local PC std and an extended morphological profile.")
PARAMS = {"n_pc_seg": 3, "scale": 10.0, "sigma": 0.5, "min_size": 5, "n_pc": 5, "grad_sigma": 1.0,
          "marker_step": 2, "compactness": 0.01, "ctx_sigma1": 1.0, "ctx_sigma2": 4.0, "std_size": 7,
          "emp_pcs": 3, "emp_radii": [1, 3, 5]}
CLASSIFIER = {"kernel": "rbf", "C": 300.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    f = np.asarray(ctx.pcs[..., : params["n_pc_seg"]], dtype=np.float64)
    a = felzenszwalb(f, scale=params["scale"], sigma=params["sigma"], min_size=params["min_size"],
                     channel_axis=-1).astype(np.int64)
    p = np.asarray(ctx.pcs[..., : params["n_pc"]], dtype=np.float64)
    g = sum(ndimage.gaussian_gradient_magnitude(p[..., i], params["grad_sigma"]) for i in range(p.shape[2]))
    st = params["marker_step"]
    m = np.zeros(g.shape, np.int32)
    yy, xx = np.mgrid[st // 2: g.shape[0]: st, st // 2: g.shape[1]: st]
    m[yy, xx] = np.arange(1, yy.size + 1).reshape(yy.shape)
    b = watershed(g, m, compactness=params["compactness"]).astype(np.int64)
    return B.relabel(a * (int(b.max()) + 1) + b)

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
