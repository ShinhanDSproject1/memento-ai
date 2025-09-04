import fitz  # PyMuPDF
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim


def detect_stamp_regions(img_bgr: np.ndarray):
    """빨간색 계열을 찾아 도장 후보 영역 반환"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 80, 60], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 80, 60], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img_bgr.shape[:2]

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (W * H) * 0.001:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        candidates.append((x, y, w, h))

    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    return candidates


def extract_and_save_stamp(
    pdf_path: str, out_path: str, page_index: int = 0, zoom: float = 2.0
) -> bool:
    """PDF → 이미지 변환 → 도장 후보 → crop → 저장"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img_data = pix.tobytes("png")

        arr = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return False

        candidates = detect_stamp_regions(img_bgr)
        if not candidates:
            return False

        x, y, w, h = candidates[0]
        crop = img_bgr[y : y + h, x : x + w]
        if crop.size == 0:
            return False

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        return cv2.imwrite(out_path, crop)
    except Exception as e:
        print("extract_and_save_stamp ERROR:", e)
        return False


# ---------- 도장 매칭 ----------
def _resize_keep_ratio(img, target_size):
    th, tw = target_size
    ih, iw = img.shape[:2]
    scale = min(tw / iw, th / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, img.shape[2]), 255, dtype=np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _ssim_score(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (3, 3), 0)
    g2 = cv2.GaussianBlur(g2, (3, 3), 0)
    h, w = min(g1.shape[0], g2.shape[0]), min(g1.shape[1], g2.shape[1])
    g1 = cv2.resize(g1, (w, h))
    g2 = cv2.resize(g2, (w, h))
    return float(max(0, min(1, ssim(g1, g2, data_range=255))))


def _orb_match_score(img1, img2):
    orb = cv2.ORB_create(1000)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if not matches:
        return 0.0
    good = [m for m in matches if m.distance < 60]
    return min(1.0, len(good) / max(len(matches), 1))


def verify_stamp_with_template(page_bgr, template_path, bbox=None):
    tpl = cv2.imread(template_path)
    if tpl is None:
        return {
            "found": False,
            "confidence": 0.0,
            "bbox": None,
            "reason": "no_template",
        }

    cand = bbox
    if cand is None:
        cands = detect_stamp_regions(page_bgr)
        if not cands:
            return {
                "found": False,
                "confidence": 0.0,
                "bbox": None,
                "reason": "no_candidate",
            }
        cand = cands[0]

    x, y, w, h = cand
    crop = page_bgr[y : y + h, x : x + w].copy()
    if crop.size == 0:
        return {"found": False, "confidence": 0.0, "bbox": None, "reason": "empty_crop"}

    th, tw = tpl.shape[:2]
    crop_n = _resize_keep_ratio(crop, (th, tw))
    conf = 0.6 * _ssim_score(crop_n, tpl) + 0.4 * _orb_match_score(crop_n, tpl)
    return {"found": True, "confidence": round(conf, 3), "bbox": [x, y, w, h]}
