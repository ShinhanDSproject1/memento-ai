import io, traceback
from typing import Dict, List, Tuple, Optional
import fitz
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pytesseract

from app.core.ocr.labels import (
    LABEL_PATTERNS_INFORPROC,
    LABEL_PATTERNS_ADSP,
    LABEL_PATTERNS_SQLD,
    normalize_text,
    compile_label_patterns,
)


# ============================================================
# PDF 렌더
# ============================================================
def render_page_png_bytes(
    pdf_bytes: bytes, page_index: int, zoom: float = 2.0
) -> bytes:
    """
    PDF 한 페이지를 zoom 비율로 렌더하여 PNG 바이트로 반환
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


# ============================================================
# 자격증 타입 감지
#  - 1차: PDF 내장 텍스트
#  - 2차(Fallback): 1페이지 OCR로 키워드 감지
# ============================================================
def detect_certificate_type_from_pdf(
    pdf_bytes: bytes,
    *,
    ocr_fallback: bool = True,
    ocr_zoom: float = 2.0,
    lang: str = "kor+eng",
) -> str:
    """
    자격증 종류 감지.
    1) PDF 내장 텍스트에서 키워드 검색
    2) (옵션) 실패 시 첫 페이지 OCR 텍스트로 보조 감지
    """
    try:
        # 1) PDF 텍스트
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

        # 2) OCR Fallback
        if ocr_fallback:
            try:
                page_png = render_page_png_bytes(pdf_bytes, page_index=0, zoom=ocr_zoom)
                arr = np.frombuffer(page_png, np.uint8)
                rgb = cv2.cvtColor(
                    cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
                cfg = "--oem 1 --psm 6"
                ocr_text = pytesseract.image_to_string(
                    rgb, lang=lang, config=cfg
                ).lower()

                if ("정보처리기사" in ocr_text) or (
                    "engineer information processing" in ocr_text
                ):
                    return "inforproc"
                if "sqld" in ocr_text:
                    return "sqld"
                if ("adsp" in ocr_text) or (
                    "advanced data analytics semi-professional" in ocr_text
                ):
                    return "adsp"
            except Exception:
                traceback.print_exc()
        return "unknown"
    except Exception:
        return "unknown"


# ============================================================
# 공통 레이아웃 추출 (기존: PDF 텍스트 파서 기반)
# ============================================================
def extract_layout_for_patterns(pdf_bytes: bytes, label_patterns: Dict[str, str]):
    """
    PDF 내장 텍스트(span) → 라벨/값 매칭
    """
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


# ============================================================
# OCR 기반: 스팬 생성 + 레이아웃 추출 (신규)
# ============================================================
def _ocr_spans_from_pdf(
    pdf_bytes: bytes, page_index: int, zoom: float = 2.0, lang: str = "kor+eng"
) -> List[dict]:
    """
    Tesseract로 spans(text + bbox in PDF coords)를 생성.
    - 페이지를 현재 zoom으로 렌더하여 OCR
    - Tesseract가 준 픽셀 좌표를 'PDF 좌표'로 환산(1/zoom)
    """
    page_png = render_page_png_bytes(pdf_bytes, page_index=page_index, zoom=zoom)
    arr = np.frombuffer(page_png, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cfg = "--oem 1 --psm 6"
    data = pytesseract.image_to_data(
        rgb, lang=lang, config=cfg, output_type=pytesseract.Output.DICT
    )

    spans: List[dict] = []
    n = len(data.get("text", []))
    for i in range(n):
        raw = data["text"][i]
        txt = (raw or "").strip()
        if not txt:
            continue
        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        # 픽셀 → PDF 좌표(렌더러가 PDF*zoom으로 그리므로 역변환)
        x0 = x / zoom
        y0 = y / zoom
        x1 = (x + w) / zoom
        y1 = (y + h) / zoom
        spans.append({"text": txt, "bbox": (x0, y0, x1, y1)})

    spans.sort(key=lambda s: (round(s["bbox"][1], 0), s["bbox"][0]))
    return spans


def extract_layout_for_patterns_ocr(
    pdf_bytes: bytes,
    label_patterns: Dict[str, str],
    page_index: int = 0,
    zoom: float = 2.0,
    lang: str = "kor+eng",
):
    """
    OCR 스팬 기반 라벨/값 매칭 (기존 로직 재사용).
    """
    compiled = compile_label_patterns(label_patterns)
    result = {"pages": 1, "fields": {}}

    spans = _ocr_spans_from_pdf(pdf_bytes, page_index=page_index, zoom=zoom, lang=lang)

    # 라벨 후보
    label_spans = []
    for s in spans:
        t = normalize_text(s["text"])
        for key, pat in compiled.items():
            if pat.search(t):
                label_spans.append({"key": key, "bbox": s["bbox"]})
                break

    # 라벨 기준으로 값 탐색
    for label in label_spans:
        lx0, ly0, lx1, ly1 = label["bbox"]
        key = label["key"]

        same_line_candidates: List[Tuple[float, dict]] = []
        below_candidates: List[Tuple[float, dict]] = []

        for s in spans:
            x0, y0, x1, y1 = s["bbox"]
            if (x0, y0, x1, y1) == (lx0, ly0, lx1, ly1):
                continue

            label_yc = (ly0 + ly1) / 2
            span_yc = (y0 + y1) / 2
            same_line = abs(span_yc - label_yc) <= 6  # 필요 시 8~10으로 조정

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
                span for _, span in sorted(same_line_candidates, key=lambda x: x[0])[:5]
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


# 자격증별 래퍼 (OCR 버전)
def extract_inforproc_ocr(
    pdf_bytes: bytes, page_index: int = 0, zoom: float = 2.0, lang: str = "kor+eng"
):
    return extract_layout_for_patterns_ocr(
        pdf_bytes, LABEL_PATTERNS_INFORPROC, page_index, zoom, lang
    )


def extract_sqld_ocr(
    pdf_bytes: bytes, page_index: int = 0, zoom: float = 2.0, lang: str = "kor+eng"
):
    return extract_layout_for_patterns_ocr(
        pdf_bytes, LABEL_PATTERNS_SQLD, page_index, zoom, lang
    )


def extract_adsp_ocr(
    pdf_bytes: bytes, page_index: int = 0, zoom: float = 2.0, lang: str = "kor+eng"
):
    return extract_layout_for_patterns_ocr(
        pdf_bytes, LABEL_PATTERNS_ADSP, page_index, zoom, lang
    )


# ============================================================
# 자격증별 래퍼 (기존: PDF 텍스트 파서)
# ============================================================
def extract_inforproc(pdf_bytes: bytes):
    return extract_layout_for_patterns(pdf_bytes, LABEL_PATTERNS_INFORPROC)


def extract_sqld(pdf_bytes: bytes):
    return extract_layout_for_patterns(pdf_bytes, LABEL_PATTERNS_SQLD)


def extract_adsp(pdf_bytes: bytes):
    return extract_layout_for_patterns(pdf_bytes, LABEL_PATTERNS_ADSP)


# ============================================================
# (신규) 자동 선택: 텍스트 우선, 실패 시 OCR 폴백
# ============================================================
def extract_layout_auto(
    pdf_bytes: bytes,
    cert_code: str,
    page_index: int = 0,
    zoom: float = 2.0,
    lang: str = "kor+eng",
):
    # 1) 텍스트 파서 우선
    if cert_code == "inforproc":
        layout = extract_inforproc(pdf_bytes)
    elif cert_code == "sqld":
        layout = extract_sqld(pdf_bytes)
    elif cert_code == "adsp":
        layout = extract_adsp(pdf_bytes)
    else:
        layout = {}

    if layout and layout.get("fields"):
        return layout

    # 2) OCR 폴백
    if cert_code == "inforproc":
        return extract_inforproc_ocr(
            pdf_bytes, page_index=page_index, zoom=zoom, lang=lang
        )
    elif cert_code == "sqld":
        return extract_sqld_ocr(pdf_bytes, page_index=page_index, zoom=zoom, lang=lang)
    elif cert_code == "adsp":
        return extract_adsp_ocr(pdf_bytes, page_index=page_index, zoom=zoom, lang=lang)
    return {}


# ============================================================
# 박스 렌더
# ============================================================
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
    """
    layout의 PDF 좌표 박스를 zoom 배율로 그려 PNG 반환.
    - value/label bbox는 PDF 좌표계임(렌더 시 *zoom)
    - OCR 경로도 bbox는 PDF 좌표로 변환해서 넣어야 함(이미 처리됨)
    """
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


# ============================================================
# 뷰어용 도장(스탬프) 단순 탐지
# ============================================================
def detect_stamp_bbox_px_from_pdf_image(
    pdf_bytes: bytes, page_index: int, zoom: float
) -> Optional[Tuple[int, int, int, int]]:
    """
    뷰어용으로 페이지 이미지를 렌더한 뒤, 이미지 좌표계에서 도장 bbox를 간단 추정하여 반환.
    반환 bbox는 '픽셀 좌표'(이미지 좌표)임을 주의.
    """
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
