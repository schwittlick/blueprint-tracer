"""Command-line entry point for the headless tracing core.

Usage:
    blueprint-tracer INPUT... -o OUTDIR [options]

INPUT may be image files or directories. For each input a ``.svg`` and ``.json``
are written to OUTDIR; ``--render`` also writes a ``_trace.png`` QA rasterization
and ``--debug`` dumps the intermediate gray/mask/skeleton images.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

from blueprint_tracer.core.config import Config
from blueprint_tracer.core.io_utils import list_images, load_gray
from blueprint_tracer.core.pipeline import run
from blueprint_tracer.export.json_export import write_json
from blueprint_tracer.export.render import render_paths
from blueprint_tracer.export.svg import write_svg


def build_config(args: argparse.Namespace) -> Config:
    """Build a Config; parameters left unset stay ``None`` so the pipeline can
    derive them from the measured stroke width."""
    cfg = Config()
    cfg.max_dim = args.max_dim
    cfg.supersample = args.supersample
    cfg.auto_scale = not args.no_auto_scale
    cfg.threshold_method = args.method
    cfg.sauvola_window = args.sauvola_window
    cfg.sauvola_k = args.sauvola_k
    cfg.rdp_epsilon = args.rdp
    cfg.spur_length = args.spur
    cfg.despeckle_min_area = args.despeckle
    cfg.close_gaps = args.close_gaps
    cfg.remove_border = args.remove_border
    cfg.fill_solid = not args.no_fill_solid
    cfg.detect_text = not args.no_detect_text
    cfg.text_gap_ratio = args.text_gap_ratio
    cfg.text_line_ratio = args.text_line_ratio
    cfg.solid_mode = args.solid_mode
    cfg.solid_min_width = args.solid_min_width
    cfg.deskew = not args.no_deskew
    cfg.flatfield = not args.no_flatfield
    cfg.ink_morph = args.ink_morph
    cfg.pre_smooth = args.pre_smooth
    cfg.join_paths = not args.no_join
    cfg.plot_order = not args.no_order
    if args.dpi:
        cfg.dpi = args.dpi
    if args.angle is not None:
        cfg.manual_angle = args.angle
    return cfg


def process_one(path: str, out_dir: str, cfg: Config, args: argparse.Namespace) -> None:
    name = os.path.splitext(os.path.basename(path))[0]
    gray, dpi = load_gray(path)
    result = run(gray, cfg, dpi=dpi, debug=args.debug)

    write_svg(result, os.path.join(out_dir, name + ".svg"))
    write_json(result, os.path.join(out_dir, name + ".json"), image_name=os.path.basename(path))

    if args.render:
        png = render_paths(result, thickness=args.render_thickness)
        cv2.imwrite(os.path.join(out_dir, name + "_trace.png"), png)
    if args.debug:
        for key, img in result.debug.items():
            out = img
            if img.dtype != "uint8":
                out = (img.astype("float32") / (img.max() or 1) * 255).astype("uint8")
            cv2.imwrite(os.path.join(out_dir, f"{name}_{key}.png"), out)

    s = result.stats
    saved = s["pen_up_before_px"] - s["pen_up_after_px"]
    print(
        f"  {os.path.basename(path)}: {result.width}x{result.height} "
        f"stroke={s['stroke_width_px']}px "
        f"-> {s['n_paths']} paths, {s['n_points']} pts, "
        f"ink={s['ink_length_px']:.0f}px, "
        f"pen-up {s['pen_up_before_px']:.0f}->{s['pen_up_after_px']:.0f}px "
        f"(saved {saved:.0f}), {s['elapsed_s']:.1f}s"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="blueprint-tracer")
    ap.add_argument("inputs", nargs="+", help="image files or directories")
    ap.add_argument("-o", "--out-dir", default="out")
    ap.add_argument("--max-dim", type=int, default=0, help="downscale longest edge to N px (0=full)")
    ap.add_argument("--supersample", type=float, default=1.0,
                    help="upscale before tracing (e.g. 2) to recover small text on low-res scans")
    ap.add_argument("--no-auto-scale", action="store_true",
                    help="disable deriving parameters from the measured stroke width")
    ap.add_argument("--method", default="sauvola", choices=["sauvola", "adaptive", "otsu"])
    ap.add_argument("--sauvola-window", type=int, default=None, help="default: auto (~4x stroke width)")
    ap.add_argument("--sauvola-k", type=float, default=0.08,
                    help="lower keeps faint/thin strokes solid; higher erodes them")
    ap.add_argument("--rdp", type=float, default=None, help="Douglas-Peucker epsilon px (default: auto)")
    ap.add_argument("--spur", type=float, default=None, help="spur pruning length px (default: auto)")
    ap.add_argument("--despeckle", type=int, default=None, help="min component area px (default: auto)")
    ap.add_argument("--close-gaps", type=int, default=0, help="gap-closing kernel (px, 0=off)")
    ap.add_argument("--remove-border", action="store_true", help="strip edge-touching scan cruft")
    ap.add_argument("--no-detect-text", action="store_true", help="skip text region detection")
    ap.add_argument("--text-gap-ratio", type=float, default=1.1,
                    help="grouping reach along a text line, in character heights "
                         "(raise if labels fragment, lower if they fuse)")
    ap.add_argument("--text-line-ratio", type=float, default=0.22,
                    help="grouping reach across text lines")
    ap.add_argument("--solid-mode", default="outline", choices=["outline", "skeleton", "ignore"],
                    help="filled shapes: trace their outline (default), skeletonize them "
                         "into a medial axis, or drop them entirely")
    ap.add_argument("--solid-min-width", type=float, default=None,
                    help="ink this wide counts as solid, not a stroke (default: auto)")
    ap.add_argument("--no-fill-solid", action="store_true",
                    help="do not restore interiors of ink wider than the threshold window")
    ap.add_argument("--dpi", type=float, default=None)
    ap.add_argument("--angle", type=float, default=None, help="manual deskew angle (deg)")
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--no-flatfield", action="store_true")
    ap.add_argument("--ink-morph", type=int, default=0,
                    help="thicken (+px) or thin (-px) the ink before binarizing; "
                         "thicken with --supersample 2 to stop thin crisp lines "
                         "skeletonizing into a wobble")
    ap.add_argument("--pre-smooth", type=float, default=0.0,
                    help="blur radius applied to the ink before binarizing (0=off)")
    ap.add_argument("--no-join", action="store_true")
    ap.add_argument("--no-order", action="store_true")
    ap.add_argument("--render", action="store_true", help="also write a QA PNG of the trace")
    ap.add_argument("--render-thickness", type=int, default=1)
    ap.add_argument("--debug", action="store_true", help="dump intermediate images")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    os.makedirs(args.out_dir, exist_ok=True)

    files: list[str] = []
    for inp in args.inputs:
        files.extend(list_images(inp))
    if not files:
        print("no input images found", file=sys.stderr)
        return 1

    print(f"tracing {len(files)} image(s) -> {args.out_dir}")
    failed = 0
    for f in files:
        try:
            process_one(f, args.out_dir, cfg, args)
        except Exception as exc:  # keep batch going; report the failure
            failed += 1
            print(f"  {os.path.basename(f)}: FAILED ({exc})", file=sys.stderr)
    if failed:
        print(f"{failed} of {len(files)} image(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
