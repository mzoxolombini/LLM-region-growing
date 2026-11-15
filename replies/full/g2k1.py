import numpy as np
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from llm_mrg import blocks as B

DESCRIPTION = ("Partitional clustering: mini-batch k-means on the ten PCs, then 8-connected components "
               "of the cluster map with tiny components merged into their closest neighbour; pixels "
               "labelled by an RBF SVM on PCs and smoothed PCs with a majority vote per region.")
PARAMS = {"k": 40, "sample": 50000, "min_size": 6, "ctx_sigma": 2.0, "seed": 0}
CLASSIFIER = {"kernel": "rbf", "C": 100.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    X = np.asarray(ctx.pcs, dtype=np.float32).reshape(-1, ctx.pcs.shape[2])
    idx = ctx.rng.choice(X.shape[0], size=min(params["sample"], X.shape[0]), replace=False)
    km = MiniBatchKMeans(n_clusters=params["k"], random_state=params["seed"], n_init=3).fit(X[idx])
    cl = km.predict(X).reshape(ctx.H, ctx.W)
    lab = B.connected_components(cl, 8, ctx)
    return B.merge_small(ctx, lab, ctx.pcs, min_size=params["min_size"])


def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    s = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma"]) for i in range(p.shape[2])], -1)
    return np.concatenate([p, s], -1)
