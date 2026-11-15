"""Building-block library available to every design (Section 8.4.2).

Baseline components
  orientation_map(v)                     Eq. (6.6)  [ASSUMED: Sobel atan2]
  mrg(ctx, ...)                          MRG of Section 6.3.3 [RECONSTRUCTED]
  glcm88(ctx, regions, rows, ...)        88 GLCM features (22 features x 4 angles)
Generic region operations
  grow(ctx, feat, order, ...)            multichannel growing, seed or running-mean
  histogram_seed_order(img, cells, ...)  seed order by histogram frequency
  grid_cells(H, W, g)                    g x g grid cell map
  region_means / region_stds             per-region mean / std of any feature map
  region_adjacency(regions)              edges + shared-boundary lengths
  merge_small(ctx, regions, feat, ...)   merge small regions into most similar neighbour
  connected_components(regions)          split non-connected regions
  relabel(regions)                       relabel to 0..R-1

Blocks that form or change regions append a line to ctx.trace. The trace is how
the post-hoc rule in select.py tells MRG-family designs (LLM-MRG) from the rest
(LLM-RB). A design that writes its own growing code leaves no MRG trace and is
treated as non-MRG unless audited by hand.

Equation (6.5) and (6.6) and the exact MRG procedure are in Chapter 6, which
was not supplied. What is here follows Reddy & Reddy's description as given in
Section 8.3/8.4: seeds in histogram-frequency order inside each cell of a 3x3
grid, growth under an intensity threshold to the seed and an orientation
threshold to the seed. Check it against Section 6.3.3 and Table 6.2 before
reporting numbers; the harness check in Section 8.5.1 (Indian Pines test OA
89.82%) is the acceptance test.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage

TWO_PI = 2.0 * np.pi


def _log(ctx, msg):
    if ctx is not None and hasattr(ctx, "trace"):
        ctx.trace.append(msg)


# --------------------------------------------------------------------------
# orientation and grid
# --------------------------------------------------------------------------
def orientation_map(v: np.ndarray, smooth_sigma: float = 0.0):
    """Gradient orientation (radians, -pi..pi) and magnitude of a 2-D image."""
    img = ndimage.gaussian_filter(v.astype(np.float64), smooth_sigma) if smooth_sigma > 0 else v.astype(np.float64)
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    return np.arctan2(gy, gx), np.hypot(gx, gy)


def grid_cells(h: int, w: int, g: int = 3) -> np.ndarray:
    r = np.minimum((np.arange(h) * g) // h, g - 1)
    c = np.minimum((np.arange(w) * g) // w, g - 1)
    return (r[:, None] * g + c[None, :]).astype(np.int32)


def histogram_seed_order(img: np.ndarray, cells: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Flat pixel indices ordered by cell, then by descending frequency of the
    pixel's histogram bin within its cell, then by raster position."""
    lo, hi = float(img.min()), float(img.max())
    b = np.clip(((img - lo) / (hi - lo + 1e-12) * n_bins).astype(np.int64), 0, n_bins - 1).ravel()
    c = cells.ravel().astype(np.int64)
    n_cells = int(c.max()) + 1
    freq = np.zeros((n_cells, n_bins), np.int64)
    np.add.at(freq, (c, b), 1)
    f = freq[c, b]
    raster = np.arange(b.size)
    # lexsort: last key is primary
    return np.lexsort((raster, b, -f, c)).astype(np.int64)


# --------------------------------------------------------------------------
# numba cores
# --------------------------------------------------------------------------
@njit(cache=True)
def _angdiff(a, b):
    d = abs(a - b) % TWO_PI
    return min(d, TWO_PI - d)


@njit(cache=True)
def _mrg_core(v, theta, gmag, cells, order, t_int, t_ori, min_grad, conn8):
    h, w = v.shape
    lab = -np.ones((h, w), np.int32)
    stack = np.empty(h * w, np.int64)
    if conn8:
        dy = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
        dx = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    else:
        dy = np.array([-1, 1, 0, 0])
        dx = np.array([0, 0, -1, 1])
    nr = 0
    for k in range(order.size):
        s = order[k]
        sy, sx = s // w, s % w
        if lab[sy, sx] >= 0:
            continue
        vs, ts, gs, cs = v[sy, sx], theta[sy, sx], gmag[sy, sx], cells[sy, sx]
        lab[sy, sx] = nr
        top = 0
        stack[top] = s
        top += 1
        while top > 0:
            top -= 1
            p = stack[top]
            py, px = p // w, p % w
            for n in range(dy.size):
                qy, qx = py + dy[n], px + dx[n]
                if qy < 0 or qy >= h or qx < 0 or qx >= w:
                    continue
                if lab[qy, qx] >= 0 or cells[qy, qx] != cs:
                    continue
                if abs(v[qy, qx] - vs) > t_int:
                    continue
                # orientation constraint; skipped where gradients are too weak
                if gmag[qy, qx] >= min_grad and gs >= min_grad:
                    if _angdiff(theta[qy, qx], ts) > t_ori:
                        continue
                lab[qy, qx] = nr
                stack[top] = qy * w + qx
                top += 1
        nr += 1
    return lab


@njit(cache=True)
def _grow_core(feat, cells, order, thr, use_mean, conn8):
    h, w, c = feat.shape
    lab = -np.ones((h, w), np.int32)
    stack = np.empty(h * w, np.int64)
    ref = np.empty(c, np.float64)
    acc = np.empty(c, np.float64)
    thr2 = thr * thr
    if conn8:
        dy = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
        dx = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    else:
        dy = np.array([-1, 1, 0, 0])
        dx = np.array([0, 0, -1, 1])
    nr = 0
    for k in range(order.size):
        s = order[k]
        sy, sx = s // w, s % w
        if lab[sy, sx] >= 0:
            continue
        cs = cells[sy, sx]
        for j in range(c):
            ref[j] = feat[sy, sx, j]
            acc[j] = feat[sy, sx, j]
        cnt = 1
        lab[sy, sx] = nr
        top = 0
        stack[top] = s
        top += 1
        while top > 0:
            top -= 1
            p = stack[top]
            py, px = p // w, p % w
            for n in range(dy.size):
                qy, qx = py + dy[n], px + dx[n]
                if qy < 0 or qy >= h or qx < 0 or qx >= w:
                    continue
                if lab[qy, qx] >= 0 or cells[qy, qx] != cs:
                    continue
                d2 = 0.0
                for j in range(c):
                    t = feat[qy, qx, j] - ref[j]
                    d2 += t * t
                if d2 > thr2:
                    continue
                lab[qy, qx] = nr
                stack[top] = qy * w + qx
                top += 1
                if use_mean:
                    cnt += 1
                    for j in range(c):
                        acc[j] += feat[qy, qx, j]
                        ref[j] = acc[j] / cnt
        nr += 1
    return lab


# --------------------------------------------------------------------------
# baseline MRG
# --------------------------------------------------------------------------
def mrg(ctx, t_intensity: float = 0.05, t_orientation: float = np.pi / 4,
        n_bins: int = 256, grid: int = 3, connectivity: int = 8,
        min_grad: float = 0.0, smooth_sigma: float = 0.0) -> np.ndarray:
    """Modified region growing on ctx.v. Returns int32 region map (H, W)."""
    v = np.ascontiguousarray(ctx.v, dtype=np.float64)
    theta, gmag = orientation_map(v, smooth_sigma)
    cells = grid_cells(*v.shape, grid)
    order = histogram_seed_order(v, cells, n_bins)
    lab = _mrg_core(v, theta, gmag, cells, order, float(t_intensity), float(t_orientation),
                    float(min_grad), connectivity == 8)
    _log(ctx, f"mrg(seed_order=histogram,criterion=seed,orientation=on,grid={grid},"
              f"t_int={t_intensity:.4g},t_ori={t_orientation:.4g},bins={n_bins})")
    return lab


# --------------------------------------------------------------------------
# generic region operations
# --------------------------------------------------------------------------
def grow(ctx, feat: np.ndarray, order: np.ndarray | None = None, threshold: float = 1.0,
         criterion: str = "seed", cells: np.ndarray | None = None,
         connectivity: int = 8) -> np.ndarray:
    """Region growing on a multichannel map feat (H, W, C) under a Euclidean
    threshold to the seed ('seed') or to the running region mean ('mean')."""
    f = np.ascontiguousarray(feat if feat.ndim == 3 else feat[..., None], dtype=np.float64)
    h, w, _ = f.shape
    if cells is None:
        cells = np.zeros((h, w), np.int32)
    if order is None:
        order = np.arange(h * w, dtype=np.int64)
    lab = _grow_core(f, np.ascontiguousarray(cells, np.int32), np.asarray(order, np.int64),
                     float(threshold), criterion == "mean", connectivity == 8)
    ncell = int(cells.max()) + 1
    _log(ctx, f"grow(criterion={criterion},channels={f.shape[2]},cells={ncell},thr={threshold:.4g})")
    return lab


def relabel(regions: np.ndarray) -> np.ndarray:
    _, inv = np.unique(regions, return_inverse=True)
    return inv.reshape(regions.shape).astype(np.int32)


def connected_components(regions: np.ndarray, connectivity: int = 8, ctx=None) -> np.ndarray:
    from skimage.measure import label
    lab = label(regions + 1, background=0, connectivity=2 if connectivity == 8 else 1)
    _log(ctx, "connected_components")
    return (lab - 1).astype(np.int32)


def region_sizes(regions: np.ndarray) -> np.ndarray:
    return np.bincount(regions.ravel())


def region_means(feat: np.ndarray, regions: np.ndarray, rows=None) -> np.ndarray:
    f = feat.reshape(regions.size, -1).astype(np.float64)
    r = regions.ravel()
    n = int(r.max()) + 1
    cnt = np.bincount(r, minlength=n).astype(np.float64)
    out = np.zeros((n, f.shape[1]))
    np.add.at(out, r, f)
    out /= np.maximum(cnt, 1)[:, None]
    return out if rows is None else out[np.asarray(rows)]


def region_stds(feat: np.ndarray, regions: np.ndarray, rows=None) -> np.ndarray:
    f = feat.reshape(regions.size, -1).astype(np.float64)
    r = regions.ravel()
    n = int(r.max()) + 1
    cnt = np.maximum(np.bincount(r, minlength=n), 1).astype(np.float64)[:, None]
    s1 = np.zeros((n, f.shape[1]))
    s2 = np.zeros((n, f.shape[1]))
    np.add.at(s1, r, f)
    np.add.at(s2, r, f * f)
    var = np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0)
    out = np.sqrt(var)
    return out if rows is None else out[np.asarray(rows)]


def region_adjacency(regions: np.ndarray):
    """Returns edges (E, 2) with a < b and the shared boundary length (E,)."""
    a = np.concatenate([regions[:, :-1].ravel(), regions[:-1, :].ravel()])
    b = np.concatenate([regions[:, 1:].ravel(), regions[1:, :].ravel()])
    m = a != b
    a, b = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
    pairs, counts = np.unique(np.stack([a, b], 1), axis=0, return_counts=True)
    return pairs.astype(np.int64), counts


def merge_small(ctx, regions: np.ndarray, feat: np.ndarray, min_size: int = 20,
                max_iter: int = 20) -> np.ndarray:
    """Merge every region smaller than min_size into its adjacent region with the
    closest mean feature vector. Repeats until no small region is left."""
    lab = relabel(regions)
    for _ in range(max_iter):
        sizes = region_sizes(lab)
        small = np.where(sizes < min_size)[0]
        if small.size == 0:
            break
        means = region_means(feat, lab)
        edges, _ = region_adjacency(lab)
        if edges.size == 0:
            break
        d = np.linalg.norm(means[edges[:, 0]] - means[edges[:, 1]], axis=1)
        target = np.arange(sizes.size)
        is_small = np.zeros(sizes.size, bool)
        is_small[small] = True
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        dd = np.concatenate([d, d])
        keep = is_small[src]
        src, dst, dd = src[keep], dst[keep], dd[keep]
        o = np.lexsort((dd, src))
        src, dst = src[o], dst[o]
        first = np.r_[True, src[1:] != src[:-1]]
        target[src[first]] = dst[first]
        # resolve chains with path compression
        for i in small:
            j = i
            seen = 0
            while target[j] != j and seen < 64:
                j = target[j]
                seen += 1
            target[i] = j
        new = relabel(target[lab])
        if new.max() == lab.max():
            break
        lab = new
    _log(ctx, f"merge_small(min_size={min_size})")
    return lab


# --------------------------------------------------------------------------
# GLCM features: 22 features x 4 angles = 88
# --------------------------------------------------------------------------
GLCM_NAMES = [
    "autocorrelation", "contrast", "correlation", "correlation_p", "cluster_prominence",
    "cluster_shade", "dissimilarity", "energy", "entropy", "homogeneity",
    "homogeneity_p", "max_probability", "sum_of_squares", "sum_average", "sum_variance",
    "sum_entropy", "difference_variance", "difference_entropy", "info_corr1", "info_corr2",
    "inverse_difference_normalized", "inverse_difference_moment_normalized",
]


@njit(cache=True)
def _glcm_accumulate(q, regions, rowmap, out, dist):
    # offsets for 0, 45, 90, 135 degrees
    h, w = q.shape
    offs = ((0, dist), (-dist, dist), (-dist, 0), (-dist, -dist))
    for y in range(h):
        for x in range(w):
            r = rowmap[regions[y, x]]
            if r < 0:
                continue
            a = q[y, x]
            for k in range(4):
                yy = y + offs[k][0]
                xx = x + offs[k][1]
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                if regions[yy, xx] != regions[y, x]:
                    continue
                b = q[yy, xx]
                out[r, k, a, b] += 1.0
                out[r, k, b, a] += 1.0  # symmetric


def _haralick22(P: np.ndarray) -> np.ndarray:
    """P: (..., L, L) normalised GLCMs. Returns (..., 22)."""
    eps = 1e-12
    L = P.shape[-1]
    i = np.arange(1, L + 1, dtype=np.float64)[:, None]
    j = np.arange(1, L + 1, dtype=np.float64)[None, :]
    px = P.sum(-1)
    py = P.sum(-2)
    mux = (px * i[:, 0]).sum(-1)
    muy = (py * j[0]).sum(-1)
    sx = np.sqrt(np.maximum((px * (i[:, 0] - mux[..., None]) ** 2).sum(-1), 0))
    sy = np.sqrt(np.maximum((py * (j[0] - muy[..., None]) ** 2).sum(-1), 0))
    sxy = sx * sy + eps
    m = lambda f: (P * f).sum((-1, -2))
    d = np.abs(i - j)
    feats = []
    feats.append(m(i * j))
    feats.append(m(d ** 2))
    feats.append((m(i * j) - mux * muy) / sxy)
    feats.append(((P * (i - mux[..., None, None]) * (j - muy[..., None, None])).sum((-1, -2))) / sxy)
    s = i + j - mux[..., None, None] - muy[..., None, None]
    feats.append((P * s ** 4).sum((-1, -2)))
    feats.append((P * s ** 3).sum((-1, -2)))
    feats.append(m(d))
    feats.append((P ** 2).sum((-1, -2)))
    ent = -(P * np.log(P + eps)).sum((-1, -2))
    feats.append(ent)
    feats.append(m(1.0 / (1.0 + d)))
    feats.append(m(1.0 / (1.0 + d ** 2)))
    feats.append(P.max((-1, -2)))
    feats.append((P * (i - mux[..., None, None]) ** 2).sum((-1, -2)))
    # sum and difference distributions
    ks = np.arange(2, 2 * L + 1)
    psum = np.stack([(P * ((i + j) == k)).sum((-1, -2)) for k in ks], -1)
    kd = np.arange(0, L)
    pdif = np.stack([(P * (d == k)).sum((-1, -2)) for k in kd], -1)
    savg = (psum * ks).sum(-1)
    feats.append(savg)
    feats.append((psum * (ks - savg[..., None]) ** 2).sum(-1))
    feats.append(-(psum * np.log(psum + eps)).sum(-1))
    dmean = (pdif * kd).sum(-1)
    feats.append((pdif * (kd - dmean[..., None]) ** 2).sum(-1))
    feats.append(-(pdif * np.log(pdif + eps)).sum(-1))
    hx = -(px * np.log(px + eps)).sum(-1)
    hy = -(py * np.log(py + eps)).sum(-1)
    pxpy = px[..., :, None] * py[..., None, :]
    hxy1 = -(P * np.log(pxpy + eps)).sum((-1, -2))
    hxy2 = -(pxpy * np.log(pxpy + eps)).sum((-1, -2))
    feats.append((ent - hxy1) / (np.maximum(hx, hy) + eps))
    feats.append(np.sqrt(np.clip(1 - np.exp(-2 * (hxy2 - ent)), 0, None)))
    feats.append(m(1.0 / (1.0 + d / L)))
    feats.append(m(1.0 / (1.0 + (d / L) ** 2)))
    return np.stack(feats, -1)


def glcm88(ctx, regions: np.ndarray, rows, levels: int = 16, distance: int = 1,
           image: np.ndarray | None = None, chunk: int = 4096) -> np.ndarray:
    """88 GLCM features (22 Haralick/Soh/Clausi features x 4 angles) per region,
    computed on pixels inside each region only. Returns (len(rows), 88).

    ASSUMPTION: 16 grey levels, distance 1, symmetric GLCM, angles 0/45/90/135.
    """
    img = ctx.v if image is None else image
    lo, hi = float(img.min()), float(img.max())
    q = np.clip(((img - lo) / (hi - lo + 1e-12) * levels).astype(np.int64), 0, levels - 1)
    rows = np.asarray(rows, np.int64)
    nreg = int(regions.max()) + 1
    out = np.zeros((rows.size, 88), np.float64)
    reg = np.ascontiguousarray(regions, np.int64)
    for s in range(0, rows.size, chunk):
        sub = rows[s:s + chunk]
        rowmap = -np.ones(nreg, np.int64)
        rowmap[sub] = np.arange(sub.size)
        G = np.zeros((sub.size, 4, levels, levels), np.float64)
        _glcm_accumulate(q, reg, rowmap, G, distance)
        tot = G.sum((-1, -2), keepdims=True)
        P = G / np.where(tot > 0, tot, 1.0)
        f = _haralick22(P)  # (n, 4, 22)
        f[(tot[..., 0, 0] == 0)] = 0.0
        out[s:s + sub.size] = f.transpose(0, 2, 1).reshape(sub.size, 88)  # feature-major
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
