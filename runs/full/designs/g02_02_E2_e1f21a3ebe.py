"""E2 of g01_03 (MRG regions + pixel vote; strong everywhere, F 95.9-97.6) and g01_01 (watershed,
best PU/HU but IP F=86 because 25-pixel regions straddle small fields). Watershed regions made
smaller (marker step 3) and labelled with g01_03's pixel vote on PCs and smoothed PCs."""
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from llm_mrg import blocks as B

DESCRIPTION = ("Compact watershed on the PC gradient with a 3-pixel marker lattice, labelled by "
               "pixel vote of an RBF SVM on PCs and smoothed PCs.")
PARAMS = {"n_pc": 5, "grad_sigma": 1.0, "marker_step": 3, "compactness": 0.01, "ctx_sigma": 2.0}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    p = np.asarray(ctx.pcs[..., : params["n_pc"]], dtype=np.float64)
    g = sum(ndimage.gaussian_gradient_magnitude(p[..., i], params["grad_sigma"]) for i in range(p.shape[2]))
    st = params["marker_step"]
    markers = np.zeros(g.shape, np.int32)
    yy, xx = np.mgrid[st // 2: g.shape[0]: st, st // 2: g.shape[1]: st]
    markers[yy, xx] = np.arange(1, yy.size + 1).reshape(yy.shape)
    return B.relabel(watershed(g, markers, compactness=params["compactness"]) - 1)


def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    s = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma"]) for i in range(p.shape[2])], -1)
    return np.concatenate([p, s], -1)
