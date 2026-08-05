# blueprint-tracer

Convert scanned mechanical construction blueprints (JPG/PNG/TIFF) into clean
**centerline vectors** for pen plotting, redrawing, and analysis.

An interactive Python desktop app: load a scan, tune the tracing with live
before/after preview, hand-fix the result with built-in vector editing, and
export to **SVG** and a **raw-geometry JSON** of polylines.

---

## Confirmed scope

| Decision | Choice |
|---|---|
| Output formats | SVG + raw geometry JSON (polylines) |
| What to extract | Everything, as-is — faithful trace, no classification/OCR |
| Trace style | Centerline (each ink stroke → one path; width stored as an attribute) |
| Curve model | Polylines only (straight segments) |
| Input quality | Mixed — clean scans, photos, and degraded/aged originals |
| Primary use | **Pen plotting** → join strokes, minimize pen-up travel, never fill |
| App type | Interactive GUI, Python desktop |
| Editing depth | Parameter tuning **+ vector editing** (delete/join/straighten/nodes) |
| Preprocessing | Adaptive threshold + auto-deskew (perspective/denoise as follow-ups) |
| Scale | Moderate — up to ~5000 px long edge, one drawing at a time |
| Geometry format | JSON (rich, with per-path metadata) |

Sample inputs in `data/` (MT-1.02 … MT-1.04): 1960s H&K drawings, 1654×1169 to
3307×4675, with cross-hatching, dimensioning, dash-dot centerlines, dense
text, and title blocks — used as the working test set.

---

## Status

**Milestone 1 (headless core + CLI) — built and working.** Traces all three
`data/` samples faithfully (geometry, hatching, dimension lines, text, title
blocks) into SVG + JSON. Full resolution 15.5 MP (MT-1.03, 3307×4675) runs in
~4 s; plot-ordering cuts pen-up travel by ~99 %. Next: GUI (Milestones 3–4).

### Quickstart (with [uv](https://docs.astral.sh/uv/))

The repo pins Python 3.12 via `.python-version` (the system 3.12 → 3.14 range
has no OpenCV/scikit-image wheels yet). `uv run` reads that file plus
`pyproject.toml`, builds the environment on first use, and runs inside it — no
manual venv or activation needed.

```bash
# run the tests
uv run --extra dev pytest -q

# trace a folder of scans -> SVG + JSON (+ a QA PNG of the trace) in ./out
uv run blueprint-tracer data/ -o out --render

# a single file, downscaled for a quick preview
uv run blueprint-tracer data/MT-1.02.tif -o out --max-dim 2000 --render
```

`uv run <cmd>` re-syncs the environment first, so it always matches
`pyproject.toml`. The first run downloads Python 3.12 and the dependencies; later
runs are instant.

**Useful flags:** `--max-dim N` (downscale for speed), `--supersample 2` (upscale
before tracing; recovers small text on low-resolution scans), `--method
sauvola|adaptive|otsu`, `--sauvola-k` (lower keeps faint strokes solid),
`--rdp` / `--spur` / `--despeckle` (auto by default — pass a value to pin it),
`--no-auto-scale`, `--no-deskew`, `--no-order`, `--angle DEG` (manual deskew),
`--debug` (dump the gray/mask/skeleton images). Full list:
`uv run blueprint-tracer --help`.

### Parameters scale to the drawing

Every length-like setting is really a multiple of the drawing's stroke width, so
the pipeline measures that width and derives the rest (threshold window, spur
length, RDP epsilon, despeckle area). A fixed `spur_length = 6 px` silently ate
the crossbars off 2 px-stroke lettering — small text traced as `(,-(YIVI II`
instead of `GROOVED CASING`. Pass any parameter explicitly to override the
derived value, or `--no-auto-scale` to fall back to fixed defaults. The resolved
values are reported in each JSON under `stats.resolved`.

<details>
<summary>Alternative: an explicit, persistent venv</summary>

```bash
uv venv                      # creates .venv on Python 3.12 (from .python-version)
uv pip install -e '.[dev]'   # editable install + pytest
uv run pytest -q
uv run blueprint-tracer data/ -o out --render
# or call the interpreter directly: .venv/bin/python -m blueprint_tracer.cli ...
```
</details>

---

## The vectorization pipeline

The core is a headless, testable library. Each stage has tunable parameters
surfaced in the GUI.

**0. Load & normalize** — read via Pillow (robust for RGBA/multi-mode TIFF),
flatten alpha over white, to grayscale. Keep original resolution and DPI (for
an optional pixel→mm scale).

**1. Preprocess**
- *Polarity* — detect ink-dark-on-light vs. blueprint light-on-dark; normalize
  to dark-on-light.
- *Flat-field* — divide by a heavily blurred copy to correct uneven
  lighting / aged-paper gradients.
- *Deskew* — estimate skew from the dominant line-angle histogram (Hough) and
  rotate; always allow a manual angle override.

**2. Binarize** — adaptive threshold, **Sauvola** by default (robust for faded
document lines), with adaptive-Gaussian and Otsu fallbacks. → binary ink mask.

**3. Cleanup** — despeckle (drop connected components below a min area),
optional gap-closing morphology (off by default, so nearby lines don't merge),
optional black scan-border removal.

**4. Skeletonize (centerline)** — `skimage` skeletonize → 1-px centerlines;
`medial_axis` distance transform gives per-pixel stroke radius → **stroke width
per path** for the JSON/SVG.

**5. Skeleton → polylines** — build the skeleton pixel graph, find endpoints
(deg 1) and junctions (deg ≥3), trace deg-2 chains into ordered polylines
(`sknw`-style). *Spur pruning* removes short dangles from junctions and
arrowhead tips (tunable length).

**6. Simplify** — Douglas–Peucker (RDP) per polyline; big point-count drop,
corners preserved (tunable epsilon).

**7. Plotter optimization** *(your primary use)*
- *Join* paths sharing an endpoint within a snap tolerance and near-collinear →
  longer continuous strokes, fewer pen lifts.
- *Order* paths with a KDTree greedy nearest-neighbour tour (reversing paths as
  needed) + optional light 2-opt → minimize pen-up travel. Reports travel saved.
- *(Nice-to-have)* merge dash-dot dashes along a common line into one path.

**8. Export**
- **SVG** — one `<path>` per stroke, `fill:none`, `stroke:#000`, stroke-width =
  measured width; draw order = plot order; units in px (or mm if DPI known).
- **JSON** — as below.

### JSON schema
```json
{
  "image": "MT-1.02.tif",
  "width_px": 1654, "height_px": 1169, "dpi": 300,
  "px_per_mm": 11.81,
  "paths": [
    { "id": 0, "points": [[102.5,340.0],[980.2,340.1]],
      "stroke_width_px": 3.2, "length_px": 1240.7,
      "closed": false, "plot_order": 0 }
  ]
}
```

---

## Tech stack

Python 3.11+ · **PySide6** (Qt GUI) · **OpenCV** + **NumPy** + **scikit-image**
(imaging/skeleton) · **SciPy** (distance transform, KDTree) · **sknw**
(skeleton graph) · **Pillow** (TIFF/RGBA I/O). Optional **Shapely** for geometry
ops. Packaged with `pyproject.toml`.

## Module layout

```
blueprint_tracer/
  core/
    io.py           load images/DPI, write SVG+JSON
    preprocess.py   polarity, flat-field, deskew
    binarize.py     sauvola / adaptive / otsu
    cleanup.py      despeckle, border, morphology
    skeleton.py     skeletonize + medial-axis width
    trace.py        skeleton graph → polylines, spur prune
    simplify.py     RDP
    optimize.py     join + plot-order
    geometry.py     Path/Polyline types
    pipeline.py     stage orchestration + Config dataclass
  gui/
    main_window.py  canvas.py  params_panel.py
    edit_tools.py   worker.py  project.py
  tests/
  pyproject.toml
```

The `core` library is fully headless (a thin CLI runs it on `data/` for
testing), so the algorithm is provable before any UI exists, and a batch CLI
stays possible later.

## GUI design

- **Canvas** (`QGraphicsView`): source layer + vector overlay, pan/zoom,
  layer toggle, overlay or side-by-side.
- **Parameters dock**: grouped controls (Preprocess / Binarize / Skeleton /
  Simplify / Plot). Every change re-runs the pipeline on a **downscaled
  preview** on a worker thread (debounced); full resolution only on export.
- **Editing tools**: select, delete, join, split, straighten (line-fit),
  node edit (drag/insert/delete vertices), reverse. Undo/redo stack.
- **Status**: path count, point count, estimated pen-up travel, timing.
- **Project files**: save/load image path + parameters + manual edits so work
  is resumable.

---

## Build milestones

1. **Headless core** — load → binarize → skeleton → trace → simplify →
   SVG/JSON, plus a CLI to run on the three `data/` samples and eyeball output.
   *De-risks the whole project first.*
2. **Plotter optimization** — join + ordering + stroke-width measurement.
3. **GUI shell** — canvas, open image, live preview + parameter panel on a
   worker thread.
4. **Vector editing** — edit tools, undo, project save/load.
5. **Polish** — deskew/perspective refinement, parameter presets, packaging;
   optional batch mode.

## Known risks & handling

- **Path explosion** from text + cross-hatching (could be tens of thousands of
  paths): spur pruning, min-length filter, RDP, and an aggressiveness preset;
  ordering stays near-linear (KDTree greedy, not O(n²)). Faithful "as-is"
  tracing keeps them, but counts are always reported.
- **Filled arrowheads** centerline to a spike → spur pruning; small and
  acceptable for a faithful trace.
- **Faded / broken lines** → optional gap-closing morphology, traded off
  against merging neighbours.
- **Deskew on dense drawings** → robust angle histogram + manual override.
- **Large scans (4675 px)** → full pipeline off the UI thread; preview
  downsampled.
