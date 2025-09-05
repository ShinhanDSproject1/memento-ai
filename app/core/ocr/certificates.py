import re, io, traceback
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import fitz
from PIL import Image, ImageDraw
import numpy as np
import cv2

from app.core.ocr.labels import (
    LABEL_PATTERNS_INFORPROC,
    LABEL_PATTERNS_ADSP,
    LABEL_PATTERNS_SQLD,
    REQUIRED_FIELDS,
    normalize_text,
    compile_label_patterns,
)


# --- PDF 렌더 ---
def render_page_png_bytes(
    pdf_bytes: bytes, page_index: int, zoom: float = 2.0
) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


# --- 자격증 감지 ---
def detect_certificate_type_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = []
        for page in doc:
            try:
                full_text.append(page.get_text("text"))
            except Exception:
                pass
        text = "\n".join(full_text).lower()
        if ("정보처리기사" in text) or ("engineer information processing" in text):
            return "inforproc"
        if "sqld" in text:
            return "sqld"
        if ("adsp" in text) or ("advanced data analytics semi-professional" in text):
            return "adsp"
        return "unknown"
    except Exception:
        return "unknown"


# --- 공통 레이아웃 추출 ---
def extract_layout_for_patterns(pdf_bytes: bytes, label_patterns: Dict[str, str]):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    compiled = compile_label_patterns(label_patterns)

    result = {"pages": len(doc), "fields": {}}

    for page_index, page in enumerate(doc):
        d = page.get_text("dict")
        spans = []
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if not txt:
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    spans.append({"text": txt, "bbox": (x0, y0, x1, y1)})

        spans.sort(key=lambda s: (round(s["bbox"][1], 0), s["bbox"][0]))

        label_spans = []
        for s in spans:
            t = normalize_text(s["text"])
            for key, pat in compiled.items():
                if pat.search(t):
                    label_spans.append({"key": key, "bbox": s["bbox"]})
                    break

        for label in label_spans:
            lx0, ly0, lx1, ly1 = label["bbox"]
            key = label["key"]

            same_line_candidates = []
            below_candidates = []

            for s in spans:
                x0, y0, x1, y1 = s["bbox"]
                if (x0, y0, x1, y1) == (lx0, ly0, lx1, ly1):
                    continue

                label_yc = (ly0 + ly1) / 2
                span_yc = (y0 + y1) / 2
                same_line = abs(span_yc - label_yc) <= 6

                if same_line and x0 > lx1:
                    dist = (x0 - lx1) + abs(span_yc - label_yc) * 0.1
                    same_line_candidates.append(
                        (dist, {"text": s["text"], "bbox": [x0, y0, x1, y1]})
                    )
                elif y0 >= ly1 and (x0 >= lx0 - 10 and x1 <= lx1 + 400):
                    dist = (y0 - ly1) + abs(x0 - lx0) * 0.05
                    below_candidates.append(
                        (dist, {"text": s["text"], "bbox": [x0, y0, x1, y1]})
                    )

            value_spans = []
            if same_line_candidates:
                value_spans = [
                    span
                    for _, span in sorted(same_line_candidates, key=lambda x: x[0])[:5]
                ]
            elif below_candidates:
                value_spans = [
                    span for _, span in sorted(below_candidates, key=lambda x: x[0])[:5]
                ]

            if value_spans and key not in result["fields"]:
                result["fields"][key] = {
                    "page_index": page_index,
                    "label_bbox": [lx0, ly0, lx1, ly1],
                    "value_spans": value_spans,
                }
    return result


# --- 자격증별 래퍼 ---
def extract_inforproc(pdf_bytes: bytes):
    return extract_layout_for_patterns(pdf_bytes, LABEL_PATTERNS_INFORPROC)


def extract_sqld(pdf_bytes: bytes):
    return extract_layout_for_patterns(pdf_bytes, LABEL_PATTERNS_SQLD)


def extract_adsp(pdf_bytes: bytes):
    return extract_layout_for_patterns(pdf_bytes, LABEL_PATTERNS_ADSP)


# --- 박스 렌더 ---
def render_page_with_boxes(
    pdf_bytes: bytes,
    layout: dict,
    page_index: int,
    zoom: float = 2.0,
    draw_label=True,
    draw_value=True,
    *,
    stamp_bbox_px: Optional[Tuple[int, int, int, int]] = None,
) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_index < 0 or page_index >= len(doc):
        raise ValueError("page out of range")

    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    draw = ImageDraw.Draw(img)
    red = (255, 0, 0)
    green = (0, 200, 0)

    for key, info in layout.get("fields", {}).items():
        if info.get("page_index") != page_index:
            continue
        if draw_label and info.get("label_bbox"):
            x0, y0, x1, y1 = info["label_bbox"]
            draw.rectangle(
                [x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom], outline=red, width=4
            )
        if draw_value:
            for vs in info.get("value_spans", []):
                bx0, by0, bx1, by1 = vs["bbox"]
                draw.rectangle(
                    [bx0 * zoom, by0 * zoom, bx1 * zoom, by1 * zoom],
                    outline=green,
                    width=3,
                )

    if stamp_bbox_px is not None:
        sx0, sy0, sx1, sy1 = stamp_bbox_px
        draw.rectangle([sx0, sy0, sx1, sy1], outline=green, width=5)  # 도장: 초록색

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# --- 뷰어용 단순 탐지 ---
def detect_stamp_bbox_px_from_pdf_image(
    pdf_bytes: bytes, page_index: int, zoom: float
) -> Optional[Tuple[int, int, int, int]]:
    try:
        from app.core.ocr.stamp import detect_stamp_bbox

        page_png = render_page_png_bytes(pdf_bytes, page_index=page_index, zoom=zoom)
        arr = np.frombuffer(page_png, np.uint8)
        page_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if page_bgr is None:
            return None
        return detect_stamp_bbox(page_bgr)
    except Exception:
        traceback.print_exc()
        return None
