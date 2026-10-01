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
| What to extract | Everything, as-is — faithful trace by default; OCR/Hershey opt-in per region |
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

**Headless core, CLI, GUI and the optional text layer are all built and working.**
Traces the `data/` samples faithfully (geometry, hatching, dimension lines, text,
title blocks) into SVG + JSON + PNG. Full resolution 15.5 MP (MT-1.03, 3307×4675)
runs in ~4 s; plot-ordering cuts pen-up travel by ~99 %. Detected lettering can be
OCR'd and re-drawn in a single-stroke Hershey font, per region, under review.

### The GUI

```bash
uv run blueprint-tracer-gui                 # or pass an image to open it directly
uv run blueprint-tracer-gui data/MT-1.02.tif
```

- **Live preview** — every parameter change re-traces on a background thread
  (debounced, downscaled to 1600 px). `Ctrl+R` re-traces at full resolution.
- **Auto parameters** — fields marked *Auto* are derived from the measured stroke
  width and shown greyed; untick to pin your own value.
- **Canvas** — scroll to zoom, middle-drag to pan, `Ctrl+0` to fit.
- **Side by side** (`Ctrl+B`) — the scan and the trace in two panes whose zoom and
  scroll stay locked together, so you can compare a detail at any magnification.
  In the single-pane view the *Image* and *Vectors* toggles overlay the two
  instead.
- **Editing** — click or rubber-band to select (shift extends), then *Delete*
  (`Del` or `Backspace`), *Join* (`J`), or *Straighten* (`T`). Tick *Nodes* to drag
  individual vertices. Full undo/redo (`Ctrl+Z` / `Ctrl+Shift+Z`). The edit keys act
  on the canvas, so a spin box in the Parameters dock keeps them while you are
  typing a value into it — click the drawing to hand focus back.
- **Text regions** — detected lettering is listed in its own dock with a crop
  preview per region. *Add text region* (`R`) drags a box around a label the
  detector missed. See [Text and OCR](#text-and-ocr) below.
- **Projects** — save the image reference, parameters, your manual edits *and*
  your text-region decisions to a `.btproj` so a session is resumable;
  re-tracing warns before discarding edits.
- **Export** — `Ctrl+E` writes `name.svg`, `name.json` and `name_trace.png`
  (a raster of the trace) exactly as shown, edits and text decisions included.

### Text and OCR

Lettering is detected automatically and listed in the **Text regions** dock. Each
region shows a crop of the scan, the recognized text, a confidence score, and what
to do with it:

| Mode | Result |
|---|---|
| `trace` (default) | the faithful centerline trace of the original lettering |
| `hershey` | a single-stroke Hershey rendering of the text, fitted to the region |
| `hide` | the region is left out of the view and the export |

Press **Run OCR** to recognize every region (Tesseract; pick the language, e.g.
`deu` for the German sheets). Recognition never changes what is drawn — it only
fills in the text, so you can read and correct it first. Text cells are editable
and re-render immediately. **Hershey if confident** switches the regions above
75 % confidence, leaving the rest traced; low-confidence rows are tinted so review
lands where the recognizer was least sure.

#### Adding a region by hand

The detector is deliberately conservative, so some labels are never claimed: an
isolated word, lettering wound into the line-work, a size far off the page's
dominant one. Press **Add text region** (`R`) and drag a box around it. The box
appears dashed on the canvas and marked ✎ in the dock, and from there it behaves
like any other region — recognize it, letter it in Hershey, or hide it. Its
orientation is read from the box's shape, so a tall narrow box letters bottom-up.
`Esc` (or `R` again) goes back to selecting strokes; while region mode is on the
left button only draws boxes, so nothing can be selected or deleted.

That replaces deleting the strokes by hand: the traced lettering stays in the
document and comes back if you switch the region to `trace`. **Remove region**
drops a box you no longer want (a detected one returns on the next re-trace).

Hand-drawn regions survive a re-trace, including one that changes the page's pixel
size — supersample, or the jump from the 1600 px preview to a full-resolution
trace — because no detection would ever put them back.

Why Hershey: these fonts are defined as pen strokes rather than filled outlines, so
a plotter draws each glyph in a single pass. A centerline trace of 2 px lettering is
always a wobbly approximation; a Hershey glyph is exactly what a pen can draw.

Nothing is destructive. Hidden and Hershey-substituted regions keep their traced
strokes in the document and are only skipped when drawing and exporting, so
switching a region back to `trace` restores it exactly. A region set to `hershey`
with no text yet keeps tracing rather than blanking the label; `hide` is the way
to say that on purpose. Text and region decisions also survive a re-trace and a
project round-trip.

**Requires**: `uv pip install -e '.[ocr]'` plus the Tesseract program and language
data (Arch: `sudo pacman -S tesseract tesseract-data-eng tesseract-data-deu`).
Without them the panel still works for detection, hiding and manual text entry.

Expect handwriting, symbols (⌀ ± ° ▽) and ink-damaged labels to stay traced — that
is the intended outcome, not a failure. A misread dimension looks authoritative in
a way a wobbly trace does not, which is why every region starts at `trace` and
Hershey is always opt-in.

#### Tuning detection, and what tuning cannot fix

Two knobs shape how glyphs are grouped into regions (`--text-gap-ratio`,
`--text-line-ratio`, or the *Text detection* group in the GUI):

- **Gap ratio** — how far grouping reaches along a line, in character heights.
  Raise it when one label fragments (`C-294-` and `127` as separate regions);
  lower it when neighbouring labels fuse into one.
- **Line ratio** — reach across lines. Raise to merge the rows of a multi-line
  label; too high and separate lines run together.

Recognition quality, though, is set mostly by the **lettering style**, and no
parameter changes that. Measured across the sample sheets: printed roman type
(`plate1`) reads at ~84 % mean confidence, while hand-lettered engineering stencil
(`m4_survival_rifle`) reads at ~34 % — at the *same* 4 px stroke width and a larger
cap height. Tesseract's models are trained on printed text; hand lettering is
effectively a different domain.

Things that do **not** rescue it, all measured rather than assumed: supersampling
(2×/3× found *fewer* regions and ran 5× slower), a character whitelist (confidence
fell to ~21 %), the LSTM-only engine mode, OCR crop upscaling from 56 to 150 px,
and pre-binarizing the crop. None moved mean confidence above ~40 %.

Raising the gap ratio does not help such sheets either — merging fragments costs
more at the recognizer than the fragments did (`m4` fell from 15 to 10 complete
part codes going from 1.1 to 2.2).

So for hand-lettered sheets the workflow is the review table, not the parameters:
detection still finds the labels, low-confidence rows are tinted, you correct the
text by hand and then switch those regions to Hershey. Anything you do not want to
correct simply stays traced. The only real fix for the recognizer itself is a model
trained on this lettering, or a higher-resolution source scan — upscaling a 14 px
glyph adds no information the recognizer did not already have.

The canvas displays the *preprocessed* page rather than the raw scan, because
deskew and preview downscaling mean only that frame shares coordinates with the
traced paths. `TraceResult` geometry is likewise in processed pixels, with `dpi`
rescaled to match so millimetre values stay correct.

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
before tracing; recovers small text on low-resolution scans), `--ink-morph 1` /
`--pre-smooth 1` (thicken and soften the ink so thin crisp lines stop
skeletonizing into a wobble — use with `--supersample 2`), `--method
sauvola|adaptive|otsu`, `--sauvola-k` (lower keeps faint strokes solid),
`--solid-mode outline|skeleton|ignore` (see below),
`--rdp` / `--spur` / `--despeckle` / `--solid-min-width` (auto by default — pass a
value to pin it), `--no-auto-scale`, `--no-deskew`, `--no-order`, `--angle DEG`
(manual deskew), `--debug` (dump the gray/mask/skeleton images). Full list:
`uv run blueprint-tracer --help`.

### Solid shapes are outlined, not skeletonized

A filled shape — an arrowhead, a blacked-out label, an ink blot — has no
meaningful centerline. Skeletonizing one yields its *medial axis*: a branching
squiggle that looks nothing like the shape. Ink wider than `solid_min_width`
(auto: 4× the stroke width, and never more than 5 % of the page) is therefore
split off and its boundary traced instead, while the surrounding line-work is
still centerlined — an arrow keeps a centerlined shaft and an outlined head.

`--solid-mode` picks the treatment: `outline` (default), `skeleton` (the medial
axis), or `ignore` (drop filled areas entirely, handy for blots and stains).

### Parameters scale to the drawing

Every length-like setting is really a multiple of the drawing's stroke width, so
the pipeline measures that width and derives the rest (threshold window, spur
length, RDP epsilon, despeckle area). A fixed `spur_length = 6 px` silently ate
the crossbars off 2 px-stroke lettering — small text traced as `(,-(YIVI II`
instead of `GROOVED CASING`. Pass any parameter explicitly to override the
derived value, or `--no-auto-scale` to fall back to fixed defaults. The resolved
values are reported in each JSON under `stats.resolved`.

The width is measured twice: once on the source, because preprocessing is itself
parameterised by it, and again on the preprocessed page, which is the frame every
later stage works in. Supersampling, rotation and ink shaping all change the
stroke, and an estimate that disagrees with the image detunes every derived
value. Both figures are reported (`stats.stroke_width_px` is the processed one).

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
- *Ink shaping* — optionally thicken or thin the ink (`--ink-morph`) and soften
  its edges (`--pre-smooth`) before thresholding. Thickening is the fix for a
  crisp but thin original: a 1–2 px anti-aliased line can only binarize into a
  staircase, and the skeleton then traces every step of it. Upscale with
  `--supersample 2` and thicken by 1 px and the same line becomes a ribbon whose
  medial axis is smooth — on `CN_224435201_U.png` that halves the point count
  (7335 → 3917) for the same ink length. Runs last, so the flat-field estimate
  and the skew search still see undistorted ink.

**2. Binarize** — adaptive threshold, **Sauvola** by default (robust for faded
document lines), with adaptive-Gaussian and Otsu fallbacks. → binary ink mask.

**3. Cleanup** — despeckle (drop connected components below a min area),
optional gap-closing morphology (off by default, so nearby lines don't merge),
optional black scan-border removal.

**3b. Split off solid regions** — a morphological opening finds ink that a disk of
`solid_min_width / 2` fits inside: empty for a stroke, near-complete for a blob.
Those areas are contoured (`cv2.findContours`) rather than skeletonized.

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
  "stats": { "plot_ordered": true, "n_paths": 713, "pen_up_after_px": 8016 },
  "text_regions": [
    { "id": 3, "x": 736, "y": 498, "width": 89, "height": 15,
      "orientation": "horizontal", "text": "GROOVED",
      "confidence": 87.0, "mode": "hershey" }
  ],
  "paths": [
    { "id": 341, "points": [[102.5,340.0],[980.2,340.1]],
      "stroke_width_px": 3.2, "length_px": 1240.7,
      "closed": false, "plot_order": 0, "region_id": -1 }
  ]
}
```

**`id` versus `plot_order`.** Paths are emitted in plotting sequence, so
`plot_order` is simply the array position. `id` is the stroke's identity in *trace*
order and therefore a permutation — which is what makes the ordering recoverable:
sort by `id` to get the pre-optimization order, or match strokes between two
exports of the same drawing. `stats.plot_ordered` says whether the optimization
ran at all.

(Numbering the paths *after* the reordering would make both fields the same 0..n-1
run, carrying no information — the array position restated twice.)

**The stats describe the geometry actually written.** `ink_length_px` and
`pen_up_after_px` are recomputed over the emitted path list, so they can be
verified against the file:

```python
import json, math
d = json.load(open("out.json"))
ps = [p["points"] for p in sorted(d["paths"], key=lambda p: p["plot_order"])]
dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
assert round(sum(dist(ps[i-1][-1], ps[i][0]) for i in range(1, len(ps)))) \
    == round(d["stats"]["pen_up_after_px"])
```

`pen_up_after_px` counts only the moves *between* paths. The approach from a home
position is excluded deliberately: the file records no home position, so including
it would make the figure impossible to check. Path direction is baked into
`points` — the optimizer reverses a path when that shortens the approach, and the
reversed order is what gets written.

**Ordering scope.** The GUI exports in two blocks — drawing strokes, then Hershey
lettering — each ordered across the whole block (`stats.plot_order_scope` records
this). The blocks are kept contiguous on purpose: they map to different pens, and a
globally optimal tour would interleave them and cost a carousel swap per path,
losing far more time at the carousel than the shorter travel saves.

---

## Tech stack

Python 3.10+ (the repo pins 3.12) · **PySide6** (Qt GUI) · **OpenCV** + **NumPy**
+ **scikit-image** (imaging/skeleton) · **SciPy** (distance transform, KDTree) ·
**Pillow** (TIFF/RGBA I/O). Packaged with `pyproject.toml`.

`sknw` was dropped for the skeleton graph — it pins a `numba` too old to build on
modern Python — so `core/trace.py` walks the skeleton itself. That turned out to
be worth it: it also let the walker cluster junction pixels, which 8-connectivity
otherwise splinters into a spray of micro-edges.

## Module layout

```
blueprint_tracer/
  core/
    io_utils.py     load images/DPI (RGBA/palette flattening)
    analyze.py      stroke-width estimate that auto-scales the parameters
    config.py       Config dataclass + resolve() for auto-scaled values
    preprocess.py   polarity, flat-field, deskew, supersample, ink shaping
    binarize.py     sauvola / adaptive / otsu + solid-ink fill
    cleanup.py      despeckle, border removal, morphology
    skeleton.py     skeletonize + medial-axis width
    trace.py        skeleton graph → polylines (junction clustering)
    simplify.py     RDP (closed-aware)
    optimize.py     spur prune, join, plot-order
    geometry.py     Path type
    pipeline.py     stage orchestration
  export/
    svg.py  json_export.py  render.py
  gui/
    app.py          entry point
    main_window.py  menus, actions, export, project handling
    canvas.py       zoomable image + vector overlay, hit-testing
    params_panel.py parameter dock with Auto fields
    edit_tools.py   edit operations + delta-based undo stack
    worker.py       background tracing thread
    project.py      .btproj save/load
  cli.py
tests/
```

The `core` library is fully headless (a thin CLI runs it on `data/` for
testing), so the algorithm is provable before any UI exists, and a batch CLI
stays possible later.

## Notes on the implementation

A few decisions that are easy to get wrong, recorded so they are not undone by
accident:

- **Vectors are painted by one `QGraphicsItem`, not one item per path.** A dense
  sheet is 15 k+ paths and that many scene items makes panning crawl; hit-testing
  runs against the numpy point arrays instead of the scene index.
- **Side-by-side gives each pane one layer only** — the scan pane never receives
  the paths and the trace pane never receives the pixmap, so the expensive path
  cache and the page bitmap are each built once rather than twice.
- **Undo stores deltas, not snapshots.** A snapshot of a 15 k-path sheet is
  megabytes per step.
- **The threshold window is sized from the stroke width, and solid ink is
  filled.** Sauvola hollows out any region wider than its window, so filled
  arrowheads and bars would otherwise come back as ragged accidental outlines —
  which is a thresholding artifact, not a decision. Filling them makes the
  `solid_mode` step the one place that decides how filled areas are drawn.
- **The solid threshold is capped against the page size.** A scan dominated by
  filled areas drags the stroke-width estimate up toward those areas, and a purely
  stroke-relative threshold would then classify nothing as solid.
- **Closed rings are simplified in closed mode.** Running open-mode
  Douglas–Peucker over a ring drops the closing vertex and leaves a gap that
  grows with radius.

---

## Roadmap

Done: headless core, plotter optimization, GUI shell, vector editing, and the
text layer (detection, OCR, Hershey substitution).

Next up:

1. **Region-specific tracing parameters.** Now that text regions are known, they
   can be binarized differently from the surrounding line-work — the real fix for
   sheets mixing 2 px lettering with 80 px solid fills.
2. **Better text grouping** — merge the lines of a multi-line label into one block
   so it can be reviewed and re-rendered as a unit.
3. **Polish** — parameter presets, perspective correction for photos, packaging,
   batch mode in the GUI.

## Known risks and handling

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
- **Damage in the source cannot be recovered.** Where ink has bled into a solid
  mass, a faithful tracer can only reproduce the mass —
  `data/plate1_provisional_ira_grenade_mark_7.png` has four such blots sitting on
  labels. No amount of parameter tuning (or OCR) brings those back.
