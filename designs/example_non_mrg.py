"""Example of a non-MRG design (what an E1 offspring might look like).
Not part of the method. Used by the smoke test and as a contract example.
"""
import numpy as np
from llm_mrg import blocks as B

DESCRIPTION = ("Running-mean region growing on the first five PCs with seeds ordered by "
               "local homogeneity, small regions merged into their most similar neighbour, "
               "pixel SVM on PCs with a majority vote per region.")

PARAMS = {"n_pc": 5, "threshold": 1.2, "min_size": 15, "smooth": 1.0}
CLASSIFIER = {"kernel": "rbf", "C": 10.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    from scipy import ndimage
    f = ctx.pcs[..., : params["n_pc"]]
    grad = sum(ndimage.gaussian_gradient_magnitude(f[..., i], params["smooth"]) for i in range(f.shape[2]))
    order = np.argsort(grad.ravel(), kind="stable")
    reg = B.grow(ctx, f, order=order, threshold=params["threshold"], criterion="mean")
    return B.merge_small(ctx, reg, f, min_size=params["min_size"])


def pixel_features(ctx, params):
    return ctx.pcs
