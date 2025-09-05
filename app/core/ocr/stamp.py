from typing import Optional, Tuple
import numpy as np
import cv2
from app.core.ocr.labels import STAMP_TEMPLATES
from app.stamp_verifier import detect_stamp_regions, verify_stamp_with_template
from app.core.ocr.certificates import render_page_png_bytes


def detect_stamp_bbox(page_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    cands = detect_stamp_regions(page_bgr)  # [(x,y,w,h), ...]
    if not cands:
        return None
    x, y, w, h = cands[0]
    return (int(x), int(y), int(x + w), int(y + h))


def compute_stamp_similarity_and_bbox(
    pdf_bytes: bytes,
    page_index: int,
    zoom: float,
    cert_code: str,
) -> tuple[float, Optional[Tuple[int, int, int, int]]]:
    template_path = STAMP_TEMPLATES.get(cert_code)
    if not template_path:
        return 0.0, None

    page_png = render_page_png_bytes(pdf_bytes, page_index=page_index, zoom=zoom)
    arr = np.frombuffer(page_png, np.uint8)
    page_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if page_bgr is None:
        return 0.0, None

    cands = detect_stamp_regions(page_bgr)
    bbox = cands[0] if cands else None

    try:
        vr = verify_stamp_with_template(page_bgr, template_path, bbox)
        similarity = float(vr.get("confidence", 0.0)) if vr else 0.0
        if bbox:
            x, y, w, h = bbox
            bbox_px = (int(x), int(y), int(x + w), int(y + h))
        else:
            bbox_px = None
        return similarity, bbox_px
    except Exception:
        return 0.0, None
