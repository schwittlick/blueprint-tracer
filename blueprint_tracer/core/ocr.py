"""Optional OCR for detected text regions.

Recognition runs *per region*, on an upscaled crop. Handing a whole blueprint to
Tesseract returns nothing useful: its page-layout analysis expects a document, not
labels scattered through line art. The same label cropped and enlarged reads
correctly.

Everything here is optional -- the module degrades to a clear message rather than
an exception when Tesseract or its language data is absent.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

# Small text needs enlarging before recognition; past ~4x there is no more detail
# to recover and it only costs time.
UPSCALE_TARGET_HEIGHT = 56
MAX_UPSCALE = 6.0
PAD = 4


class OcrUnavailable(RuntimeError):
    """Tesseract, its Python binding, or the requested language data is missing."""


@dataclass
class OcrResult:
    text: str
    confidence: float


def is_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def available_languages() -> list[str]:
    """Language codes Tesseract can actually load."""
    if shutil.which("tesseract") is None:
        return []
    try:
        out = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    lines = (out.stdout or "").splitlines()
    return sorted(x.strip() for x in lines[1:] if x.strip() and " " not in x.strip())


def check_ready(lang: str = "eng") -> None:
    """Raise :class:`OcrUnavailable` with actionable advice, or return quietly."""
    if shutil.which("tesseract") is None:
        raise OcrUnavailable(
            "The tesseract program was not found.\n"
            "Install it, e.g. on Arch:  sudo pacman -S tesseract"
        )
    try:
        import pytesseract  # noqa: F401
    except ImportError as exc:
        raise OcrUnavailable(
            "The pytesseract package is missing.\n"
            "Install the OCR extra:  uv pip install -e '.[ocr]'"
        ) from exc

    langs = available_languages()
    for code in lang.split("+"):
        if code not in langs:
            raise OcrUnavailable(
                f"Tesseract has no data for language {code!r}.\n"
                f"Installed: {', '.join(langs) or 'none'}\n"
                f"On Arch:  sudo pacman -S tesseract-data-{code}"
            )


def recognize_region(page: np.ndarray, region, lang: str = "eng") -> OcrResult:
    """Recognize one region from the page it was detected on."""
    import pytesseract
    from pytesseract import Output

    crop = _prepare_crop(page, region)
    if crop is None:
        return OcrResult("", -1.0)

    # psm 7 reads a single line, which is what a detected region is; psm 6 copes
    # when the line was merged from more than one row of glyphs.
    for psm in (7, 6):
        try:
            data = pytesseract.image_to_data(
                crop, lang=lang, config=f"--psm {psm}", output_type=Output.DICT
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as-is
            raise OcrUnavailable(str(exc)) from exc

        words, confidences = [], []
        for text, conf in zip(data["text"], data["conf"]):
            text = (text or "").strip()
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            if text and conf >= 0:
                words.append(text)
                confidences.append(conf)
        if words:
            return OcrResult(" ".join(words), float(np.mean(confidences)))
    return OcrResult("", -1.0)


def recognize_regions(page: np.ndarray, regions: list, lang: str = "eng") -> int:
    """Fill in ``text`` and ``confidence`` on each region. Returns the number read."""
    check_ready(lang)
    found = 0
    for region in regions:
        result = recognize_region(page, region, lang=lang)
        region.text = result.text
        region.confidence = result.confidence
        if result.text:
            found += 1
    return found


def _prepare_crop(page: np.ndarray, region):
    h, w = page.shape[:2]
    y0 = max(0, region.y - PAD)
    y1 = min(h, region.y + region.height + PAD)
    x0 = max(0, region.x - PAD)
    x1 = min(w, region.x + region.width + PAD)
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None

    crop = page[y0:y1, x0:x1]
    if getattr(region, "orientation", "horizontal") == "vertical":
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    scale = min(MAX_UPSCALE, max(1.0, UPSCALE_TARGET_HEIGHT / max(1, crop.shape[0])))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # A white margin helps Tesseract far more than any thresholding here.
    return cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
