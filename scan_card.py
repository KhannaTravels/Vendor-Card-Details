
"""
Business-card OCR.

One PaddleOCR pass per image with detection settings tuned for business
cards, so small or low-contrast text is found without a separate crop pass.

Text boxes are found on a reduced copy (fast) and read at full size
(accurate). A second, detailed pass on a contrast-enhanced image runs only
when the first pass looks weak (dark / blurry photo).
"""

import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

# Longest image side used for reading the text (recognition).
OCR_MAX_SIDE = 2000

# Longest side used for finding text boxes (detection).
DET_SIDE_FAST = 960

# Used for the second pass on difficult photos only.
DET_SIDE_DETAILED = 1600

# Lines read with lower confidence than this are treated as noise.
MIN_LINE_SCORE = 0.5

# Weak-result detection settings.
WEAK_MIN_LINES = 4
WEAK_AVG_SCORE = 0.85

# OCR engine is created only when a card is actually scanned.
ocr = None

# Protect OCR initialization and prediction from concurrent requests.
_ocr_lock = threading.Lock()


def _get_ocr():
    """Create the shared OCR engine only when it is first needed."""
    global ocr

    if ocr is None:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=0.2,
            text_det_box_thresh=0.4,
        )

    return ocr


def _load_image(image):
    """Accept a file path or PIL image; return an upright RGB image."""
    if isinstance(image, Image.Image):
        img = image.copy()
    else:
        path = Path(image)

        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as original:
            img = ImageOps.exif_transpose(original)
            img.load()

    img = img.convert("RGB")
    img.thumbnail(
        (OCR_MAX_SIDE, OCR_MAX_SIDE),
        Image.Resampling.LANCZOS,
    )

    return img


def _run_ocr(img, det_side=None):
    """OCR a PIL image in memory. Returns ([(text, score)], sideways)."""

    # NumPy is required only for local OCR, not for Flask startup or SSO.
    import numpy as np

    # PaddleOCR is also loaded only when OCR is actually requested.
    det_side = det_side or DET_SIDE_FAST

    array = np.ascontiguousarray(
        np.asarray(img)[:, :, ::-1]
    )  # RGB -> BGR

    with _ocr_lock:
        results = _get_ocr().predict(
            array,
            text_det_limit_side_len=det_side,
            text_det_limit_type="max",
        )

    lines, vertical, boxes = [], 0, 0

    for page in results:
        data = page.json
        res = data.get("res", data)

        polys = res.get("rec_polys") or res.get("dt_polys") or []
        boxes += len(polys)

        for poly in polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]

            if (max(ys) - min(ys)) > 1.5 * (max(xs) - min(xs)):
                vertical += 1

        for text, score in zip(
            res.get("rec_texts", []),
            res.get("rec_scores", []),
        ):
            text = " ".join(str(text).split())

            if text and score >= MIN_LINE_SCORE:
                lines.append((text, float(score)))

    # The card looks sideways when most text boxes are tall and narrow.
    sideways = boxes > 0 and vertical >= 0.6 * boxes

    return lines, sideways


def _is_weak(lines):
    if len(lines) < WEAK_MIN_LINES:
        return True

    return sum(score for _, score in lines) / len(lines) < WEAK_AVG_SCORE


def _quality(lines):
    """More confidently-read characters = better result."""
    return sum(len(text) * score for text, score in lines)


def _enhance(img):
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.5)

    return ImageEnhance.Sharpness(img).enhance(2.0)


def extract_lines(image):
    """Return OCR lines as [(text, confidence)] for one card image."""
    img = _load_image(image)

    lines, sideways = _run_ocr(img)

    # Card photographed sideways: rotate and try again.
    if sideways:
        print("OCR: card is sideways, rotating...")

        good_first = (
            lines
            and sum(score for _, score in lines) / len(lines) >= 0.8
        )

        for angle in ((90,) if good_first else (90, -90)):
            rotated = img.rotate(angle, expand=True)
            rotated_lines, _ = _run_ocr(rotated)

            if _quality(rotated_lines) >= 0.9 * _quality(lines):
                img, lines = rotated, rotated_lines
                break

    # Try a detailed pass if the first result is weak.
    if _is_weak(lines):
        print(
            "OCR: weak result, trying a detailed pass "
            "on an enhanced image..."
        )

        enhanced, _ = _run_ocr(
            _enhance(img),
            det_side=max(DET_SIDE_DETAILED, DET_SIDE_FAST),
        )

        if _quality(enhanced) > _quality(lines):
            lines = enhanced

    return lines


def extract_text(image):
    """Return the card's text, one line per detected text line."""
    return "\n".join(
        text for text, _ in extract_lines(image)
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python scan_card.py "
            "path_to_business_card_image [...]"
        )
        raise SystemExit(1)

    from destinations import find_destination_list

    for image_path in sys.argv[1:]:
        start = time.perf_counter()
        text = extract_text(image_path)

        print(
            f"\n===== {image_path} "
            f"({time.perf_counter() - start:.1f}s)"
        )
        print(text)
        print(
            "Destinations list found:",
            find_destination_list(text) or "none",
        )
