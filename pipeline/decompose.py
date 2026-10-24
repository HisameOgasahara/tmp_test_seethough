"""Step 1: split one illustration into semantic RGBA layers with See-through.

Runs See-through's NF4 (low-VRAM) script as a subprocess, then reads the PSD it writes.
Layer order in the PSD is back -> front (See-through sorts by depth).
"""
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Layer:
    name: str          # See-through tag, e.g. "front hair", "eyel", "mouth"
    rgba: np.ndarray   # full-canvas RGBA (H, W, 4), uint8
    index: int         # draw order, 0 = backmost


def run_seethrough(image_path, seethrough_dir, out_dir, resolution=1280, quantized=True, extra_args=()):
    """Run See-through and return the path of the layered PSD it produced."""
    script = "inference_psd_quantized.py" if quantized else "inference_psd.py"
    # NAI file names are full prompts (spaces, commas, dots); give See-through a plain name
    os.makedirs(out_dir, exist_ok=True)
    safe = os.path.join(out_dir, "input" + os.path.splitext(image_path)[1].lower())
    shutil.copyfile(image_path, safe)
    cmd = [sys.executable, "-u", f"inference/scripts/{script}",
           "--srcp", safe,
           "--save_dir", os.path.abspath(out_dir),
           "--resolution", str(resolution),
           "--save_to_psd", *extra_args]
    print("[decompose]", " ".join(cmd), flush=True)
    # stream the child's output into the notebook cell (Colab hides it otherwise)
    tail = []
    with subprocess.Popen(cmd, cwd=seethrough_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as p:
        for line in p.stdout:
            print(line, end="", flush=True)
            tail = (tail + [line])[-40:]
    if p.returncode:
        raise RuntimeError(f"See-through failed (exit {p.returncode}). Last output:\n" + "".join(tail))
    return find_psd(out_dir, safe)


def find_psd(out_dir, image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    cands = [p for p in glob.glob(os.path.join(out_dir, "**", "*.psd"), recursive=True)
             if not p.endswith("_depth.psd")]
    cands.sort(key=lambda p: (stem not in os.path.basename(p), -os.path.getmtime(p)))
    if not cands:
        raise FileNotFoundError(f"no PSD found under {out_dir}")
    return cands[0]


def load_psd_layers(psd_path, min_pixels=16):
    """Read PSD pixel layers into full-canvas RGBA arrays, keeping PSD order."""
    from psd_tools import PSDImage
    psd = PSDImage.open(psd_path)
    W, H = psd.width, psd.height
    layers = []
    for lay in psd.descendants():
        if lay.is_group() or lay.kind != "pixel":
            continue
        img = lay.topil()
        if img is None:
            continue
        canvas = np.zeros((H, W, 4), np.uint8)
        part = np.asarray(img.convert("RGBA"))
        x, y = lay.left, lay.top
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + part.shape[1], W), min(y + part.shape[0], H)
        if x1 <= x0 or y1 <= y0:
            continue
        canvas[y0:y1, x0:x1] = part[y0 - y:y1 - y, x0 - x:x1 - x]
        if (canvas[..., 3] > 0).sum() < min_pixels:
            continue
        layers.append(Layer(lay.name, canvas, len(layers)))
    return layers, (W, H)


def content_box(layers, pad=8):
    """Bounding box of everything visible, used as the SVG viewBox (drops See-through's square padding)."""
    alpha = np.max(np.stack([l.rgba[..., 3] for l in layers]), axis=0)
    ys, xs = np.nonzero(alpha > 8)
    H, W = alpha.shape
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
            min(xs.max() + pad, W - 1), min(ys.max() + pad, H - 1))


def save_layers(layers, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for l in layers:
        Image.fromarray(l.rgba).save(os.path.join(out_dir, f"{l.index:02d}_{l.name.replace(' ', '_')}.png"))
