"""Step 2: upscale RGBA layers with the spandrel models from HisameOgasahara/tmp_upscaler.

Upscalers are RGB-only, so:
  * color: transparent pixels are first filled with the nearest opaque color
    (alpha bleed) so edges don't pick up a dark/black fringe;
  * alpha: the mask is upscaled through the same model as a gray image, which keeps
    edges as crisp as the color.
"""
import sys

import cv2
import numpy as np
from PIL import Image

_models = None


def load(upscaler_dir, model_dir="models", names=None):
    """Import server_core from the cloned tmp_upscaler repo and load its models once."""
    global _models
    if _models is None:
        if upscaler_dir not in sys.path:
            sys.path.insert(0, upscaler_dir)
        import server_core
        server_core.download_models(model_dir, names)
        models, device = server_core.load_models(model_dir, prefer_gpu=True)
        _models = (server_core, models, device)
        print(f"[upscale] models: {list(models)} on {device}")
    return _models


def alpha_bleed(rgba):
    rgb = rgba[..., :3].copy()
    mask = rgba[..., 3] > 0
    if mask.all() or not mask.any():
        return rgb
    # nearest-opaque fill via distance transform labels
    _, lab = cv2.distanceTransformWithLabels((~mask).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(mask)
    src_of_label = np.zeros(lab.max() + 1, np.int64)
    # each opaque pixel is its own label; transparent pixels get the label of the nearest one
    lbl_at_opaque = lab[ys, xs]
    src_of_label[lbl_at_opaque] = ys * rgb.shape[1] + xs
    flat = rgb.reshape(-1, 3)
    return flat[src_of_label[lab]].reshape(rgb.shape)


def upscale_rgba(rgba, upscaler_dir, model="4x-UltraSharpV2_Lite", tile=512, crop_box=None):
    """Upscale one RGBA layer. crop_box=(x0,y0,x1,y1) limits work to the layer's content."""
    core, models, _ = load(upscaler_dir)
    key = core.resolve_model(models, model)
    if crop_box is None:
        ys, xs = np.nonzero(rgba[..., 3] > 0)
        if len(xs) == 0:
            return None, (0, 0), 1
        pad = 8
        crop_box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                    min(xs.max() + pad + 1, rgba.shape[1]), min(ys.max() + pad + 1, rgba.shape[0]))
    x0, y0, x1, y1 = crop_box
    part = rgba[y0:y1, x0:x1]
    rgb_up, scale = core.upscale_image(models[key], Image.fromarray(alpha_bleed(part)), tile=tile)
    a = np.repeat(part[..., 3:4], 3, axis=2)
    a_up, _ = core.upscale_image(models[key], Image.fromarray(a), tile=tile)
    out = np.dstack([np.asarray(rgb_up.convert("RGB")), np.asarray(a_up.convert("L"))])
    return out, (x0, y0), scale
