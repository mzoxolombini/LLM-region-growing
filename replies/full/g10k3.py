"""M1 on g09_03. The 2-pixel watershed lattice is a fixed grid, so region shapes are partly
imposed by the lattice. Replace it with 4-pixel SLIC superpixels, which adapt their shape to the
data, intersected with the Felzenszwalb segments."""
import numpy as np
from scipy import ndimage
from skimage.segmentation import slic, felzenszwalb
from llm_mrg import blocks as B

DESCRIPTION = ("Wide-context intersection with 30 smoothed raw bands added to the pixel features: of Felzenszwalb segments on three PCs and a 2-pixel-lattice PC-gradient "
               "watershed; pixel vote of an RBF SVM on PCs, two smoothing scales and local PC std.")
PARAMS = {"n_pc_seg": 3, "scale": 10.0, "sigma": 0.5, "min_size": 5, "n_pc": 5, "grad_sigma": 1.0,
          "slic_pixels": 4, "slic_compactness": 0.1, "compactness": 0.01, "ctx_sigma1": 1.0, "ctx_sigma2": 5.0, "std_size": 9,
          "n_bands": 30, "band_sigma": 1.0}
CLASSIFIER = {"kernel": "rbf", "C": 1000.0, "gamma": "scale"}
LABELING = "pixel_vote"


def segment(ctx, params):
    f = np.asarray(ctx.pcs[..., : params["n_pc_seg"]], dtype=np.float64)
    a = felzenszwalb(f, scale=params["scale"], sigma=params["sigma"], min_size=params["min_size"],
                     channel_axis=-1).astype(np.int64)
    p = np.asarray(ctx.pcs[..., : params["n_pc"]], dtype=np.float64)
    n = max(16, int(ctx.H * ctx.W / params["slic_pixels"]))
    b = slic(p, n_segments=n, compactness=params["slic_compactness"], sigma=0.5, channel_axis=-1,
             start_label=0, enforce_connectivity=True).astype(np.int64)
    return B.relabel(a * (int(b.max()) + 1) + b)

def pixel_features(ctx, params):
    p = np.asarray(ctx.pcs, dtype=np.float32)
    k = p.shape[2]
    s1 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma1"]) for i in range(k)], -1)
    s2 = np.stack([ndimage.gaussian_filter(p[..., i], params["ctx_sigma2"]) for i in range(k)], -1)
    sz = params["std_size"]
    mu = np.stack([ndimage.uniform_filter(p[..., i], sz) for i in range(k)], -1)
    sq = np.stack([ndimage.uniform_filter(p[..., i] ** 2, sz) for i in range(k)], -1)
    bsel = np.linspace(0, ctx.B - 1, params["n_bands"]).astype(int)
    bands = np.stack([ndimage.gaussian_filter(np.asarray(ctx.bands[..., j], np.float32), params["band_sigma"])
                      for j in bsel], -1)
    return np.concatenate([p, s1, s2, np.sqrt(np.maximum(sq - mu ** 2, 0)), bands], -1)
