"""Step 3: flat-color RGBA layer -> SVG paths (+ soft-gradient overlays).

Same method as the single-image vectorize.py, made alpha-aware:
  edge-preserving smoothing -> k-means palette (+ detail pass for thin lines)
  -> majority-filter cleanup -> VTracer (stacked, spline) -> radial gradients
  for soft shading the flat fill lost.
"""
import io
import os
import re
import tempfile

import cairosvg
import cv2
import numpy as np
import vtracer
from PIL import Image

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)


def lab(rgb):
    return cv2.cvtColor(np.asarray(rgb).reshape(-1, 1, 3).astype(np.uint8),
                        cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def quantize(px, pal):
    """Nearest palette index per pixel (Lab). Solves each unique color once."""
    key = (px[:, 0].astype(np.int32) << 16) | (px[:, 1].astype(np.int32) << 8) | px[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    uc = np.stack([(uk >> 16) & 255, (uk >> 8) & 255, uk & 255], 1)
    L, P = lab(pal), lab(uc)
    best = np.empty(len(P), np.int32)
    for s in range(0, len(P), 200000):
        best[s:s + 200000] = ((P[s:s + 200000, None] - L[None]) ** 2).sum(2).argmin(1)
    return best[inv.ravel()]


def build_palette(px, k, merge_de, detail_de):
    rng = np.random.default_rng(0)
    samp = px[rng.choice(len(px), min(300000, len(px)), replace=False)].astype(np.float32)
    k = max(1, min(k, len(np.unique(samp, axis=0))))
    _, lbl, cent = cv2.kmeans(samp, k, None, CRIT, 3, cv2.KMEANS_PP_CENTERS)
    L = lab(cent)
    cnt = np.bincount(lbl.ravel(), minlength=k)
    keep = []
    for i in np.argsort(-cnt):
        if cnt[i] and all(np.linalg.norm(L[i] - L[j]) > merge_de for j in keep):
            keep.append(i)
    pal = cent[keep]
    # detail pass: k-means favors big areas; give badly-fit pixels (thin lines) their own colors
    for _ in range(2):
        err = np.linalg.norm(lab(px) - lab(pal[quantize(px, pal)]), axis=1)
        bad = px[err > detail_de].astype(np.float32)
        if len(bad) < 50:
            break
        _, _, extra = cv2.kmeans(bad, min(12, len(bad) // 50), None, CRIT, 3, cv2.KMEANS_PP_CENTERS)
        L = lab(pal)
        add = [e for e, le in zip(extra, lab(extra)) if np.linalg.norm(L - le, axis=1).min() > merge_de]
        if not add:
            break
        pal = np.vstack([pal, add])
    return pal


def majority(lbl, n, valid):
    best = np.full(lbl.shape, -1.0, np.float32)
    out = lbl.copy()
    for i in range(n):
        v = cv2.boxFilter(((lbl == i) & valid).astype(np.float32), -1, (3, 3))
        m = (v > best) & valid
        best[m] = v[m]
        out[m] = i
    return out


def trace_rgba(q_rgba, speckle):
    with tempfile.TemporaryDirectory() as d:
        ip, op = os.path.join(d, "q.png"), os.path.join(d, "q.svg")
        Image.fromarray(q_rgba).save(ip)
        vtracer.convert_image_to_svg_py(ip, op, colormode="color", hierarchical="stacked", mode="spline",
                                        filter_speckle=speckle, color_precision=8, layer_difference=1,
                                        corner_threshold=60, length_threshold=4, splice_threshold=45,
                                        path_precision=2)
        return open(op).read()


def svg_body(svg_text):
    """Inner markup of an <svg> document."""
    return re.sub(r"(?s)^.*?<svg[^>]*>|</svg>\s*$", "", svg_text).strip()


def render(body, w, h):
    doc = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">{body}</svg>'
    return np.asarray(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=doc.encode()))).convert("RGBA"))


def soft_gradients(work, rend, valid, lbl, prefix, soft, region_purity=0.85):
    """Find low-frequency residuals inside one flat region and express them as radial gradients."""
    H, W = valid.shape
    res = (work[..., :3].astype(np.float32) - rend[..., :3].astype(np.float32)) * valid[..., None]
    k = max(3, int(max(W, H) * 0.01)) | 1
    mag = np.linalg.norm(cv2.GaussianBlur(res, (0, 0), k), axis=2) * valid
    n, cc, st, _ = cv2.connectedComponentsWithStats((mag > soft).astype(np.uint8))
    out, minarea = [], (max(W, H) * 0.015) ** 2
    for i in range(1, n):
        if st[i, 4] < minarea:
            continue
        ys, xs = np.nonzero(cc == i)
        labs = lbl[ys, xs]
        if np.bincount(labs[labs >= 0]).max(initial=0) / len(labs) < region_purity:
            continue
        target = work[ys, xs, :3].astype(np.float32).mean(0)
        base = rend[ys, xs, :3].astype(np.float32).mean(0)
        wgt = mag[ys, xs] + 1e-6
        cx, cy = np.average(xs, weights=wgt), np.average(ys, weights=wgt)
        rx = max(4, 2.2 * np.sqrt(np.average((xs - cx) ** 2, weights=wgt)))
        ry = max(4, 2.2 * np.sqrt(np.average((ys - cy) ** 2, weights=wgt)))
        d = target - base
        op = float(np.clip(np.abs(d).max() / 120, 0.15, 0.8))
        c = np.clip(base + d / op, 0, 255).astype(int)
        hexc = f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}"
        gid = f"{prefix}g{i}"
        out.append(f'<radialGradient id="{gid}"><stop offset="0" stop-color="{hexc}" stop-opacity="{min(op*1.6,1):.2f}"/>'
                   f'<stop offset="1" stop-color="{hexc}" stop-opacity="0"/></radialGradient>'
                   f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="url(#{gid})"/>')
    return "".join(out)


def vectorize_rgba(rgba, prefix="L", colors=32, merge=6.0, detail=12.0, speckle=2, soft=4.0, alpha_cut=128):
    """RGBA array -> (svg inner markup in the array's pixel coords, stats dict)."""
    H, W = rgba.shape[:2]
    valid = rgba[..., 3] >= alpha_cut
    if valid.sum() < 16:
        return "", {"paths": 0}
    smooth = cv2.pyrMeanShiftFiltering(np.ascontiguousarray(rgba[..., :3]), sp=4, sr=10)
    px = smooth[valid]
    pal = build_palette(px, colors, merge, detail)
    lbl = np.full((H, W), -1, np.int32)
    lbl[valid] = quantize(px, pal)
    lbl = majority(lbl, len(pal), valid)
    q = np.zeros((H, W, 4), np.uint8)
    q[valid, :3] = pal[lbl[valid]].astype(np.uint8)
    q[valid, 3] = 255
    body = svg_body(trace_rgba(q, speckle))
    rend = render(body, W, H)
    body += soft_gradients(rgba, rend, valid, lbl, prefix, soft)
    return body, {"paths": body.count("<path"), "colors": len(pal)}
