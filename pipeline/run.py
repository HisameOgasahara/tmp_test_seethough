"""End-to-end CLI.

python -m pipeline.run IMAGE --seethrough /content/see-through --upscaler /content/tmp_upscaler \
       [--closed-eyes CLOSED.png] [--out out] [--model 4x-UltraSharpV2_Lite] [--no-upscale]
       [--from-psd LAYERS.psd]   # skip See-through and reuse a PSD you already have
"""
import argparse
import json
import os
import shutil
import time

import cairosvg

from . import assemble, decompose, vectorize


def process(layers, args, prefix, extra_class="", only_roles=None):
    groups, stats = [], []
    for l in layers:
        role = assemble.role_of(l.name)
        if only_roles and role not in only_roles:
            continue
        t = time.time()
        if args.no_upscale:
            rgba, off, scale = l.rgba, (0, 0), 1
        else:
            from . import upscale
            rgba, off, scale = upscale.upscale_rgba(l.rgba, args.upscaler, args.model, args.tile)
            if rgba is None:
                continue
        body, st = vectorize.vectorize_rgba(rgba, prefix=f"{prefix}{l.index}_", colors=args.colors,
                                            merge=args.merge, detail=args.detail, speckle=args.speckle,
                                            soft=args.soft)
        if not body:
            continue
        g = assemble.layer_group(l.name, body, off, scale, role, extra_class)
        groups.append(g)
        name = f"{prefix}{l.index:02d}_{l.name.replace(' ', '_')}"
        with open(os.path.join(args.out, "layers", name + ".svg"), "w") as f:
            f.write(assemble.build_svg([g], args.box))
        stats.append({"layer": l.name, "role": role, **st, "sec": round(time.time() - t, 1)})
        print(f"[layer] {l.name:<14} role={role:<10} paths={st['paths']:<5} {stats[-1]['sec']}s")
    return groups, stats


def decompose_image(path, args, tag):
    if args.from_psd and tag == "main":
        psd = args.from_psd
    else:
        psd = decompose.run_seethrough(path, args.seethrough, os.path.join(args.out, "seethrough", tag),
                                       resolution=args.resolution, quantized=not args.full_precision)
    layers, size = decompose.load_psd_layers(psd)
    decompose.save_layers(layers, os.path.join(args.out, "layers_png", tag))
    return layers


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--closed-eyes", help="same picture with eyes closed (NAI inpaint) for cel blink")
    ap.add_argument("--seethrough", default="/content/see-through")
    ap.add_argument("--upscaler", default="/content/tmp_upscaler")
    ap.add_argument("--out", default="out")
    ap.add_argument("--from-psd")
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--full-precision", action="store_true", help="bf16 pipeline (needs ~16GB VRAM)")
    ap.add_argument("--no-upscale", action="store_true")
    ap.add_argument("--model", default="4x-UltraSharpV2_Lite")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--colors", type=int, default=32)
    ap.add_argument("--merge", type=float, default=6.0)
    ap.add_argument("--detail", type=float, default=12.0)
    ap.add_argument("--speckle", type=int, default=2)
    ap.add_argument("--soft", type=float, default=4.0)
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(args.out, "layers"), exist_ok=True)
    t0 = time.time()

    layers = decompose_image(args.image, args, "main")
    args.box = decompose.content_box(layers)
    groups, stats = process(layers, args, "L")

    if args.closed_eyes:
        closed = decompose_image(args.closed_eyes, args, "closed")
        cg, cs = process(closed, args, "C", extra_class="closed", only_roles={"eyes"})
        # put closed-eye cels right after the last open-eye layer so they share its depth
        idx = max((i for i, g in enumerate(groups) if 'data-role="eyes"' in g), default=len(groups) - 1)
        groups = groups[:idx + 1] + cg + groups[idx + 1:]
        stats += cs

    svg = assemble.build_svg(groups, args.box)
    with open(os.path.join(args.out, "character.svg"), "w") as f:
        f.write(svg)
    with open(os.path.join(args.out, "character.html"), "w") as f:
        f.write(assemble.build_html(svg))
    x0, y0, x1, y1 = args.box
    cairosvg.svg2png(bytestring=svg.encode(), write_to=os.path.join(args.out, "preview.png"),
                     output_width=min(1600, (x1 - x0) * 2))
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump({"layers": stats, "total_sec": round(time.time() - t0, 1)}, f, ensure_ascii=False, indent=2)
    zp = shutil.make_archive(args.out.rstrip("/"), "zip", args.out)
    print(f"done in {time.time()-t0:.0f}s -> {zp}")
    return zp


if __name__ == "__main__":
    main()
