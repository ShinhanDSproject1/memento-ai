from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
import io, traceback

from app.core.ocr.certificates import (
    detect_certificate_type_from_pdf,
    extract_inforproc,
    extract_adsp,
    extract_sqld,
    render_page_with_boxes,
    render_page_png_bytes,
    detect_stamp_bbox_px_from_pdf_image,
)
from app.core.ocr.labels import ALLOWED_CERT_CODES, CERT_NAME_MAP, THRESHOLD
from app.core.ocr.stamp import compute_stamp_similarity_and_bbox
from app.core.storage.s3_uploader import upload_certification_image

router = APIRouter()


@router.post("/annotated")
async def annotated(
    file: UploadFile = File(...),
    page: int = Query(0, ge=0),
    zoom: float = Query(2.0, gt=0, le=6.0),
    labels: bool = Query(True, description="라벨 박스 표시"),
    values: bool = Query(True, description="값 박스 표시"),
    show_stamp: bool = Query(True, description="도장 박스 표시"),
):
    try:
        content = await file.read()
        cert = detect_certificate_type_from_pdf(content)

        if cert == "inforproc":
            layout = extract_inforproc(content)
        elif cert == "adsp":
            layout = extract_adsp(content)
        elif cert == "sqld":
            layout = extract_sqld(content)
        else:
            return JSONResponse(
                {"error": "지원하지 않는 자격증입니다.", "detected": cert},
                status_code=400,
            )

        stamp_bbox_px = None
        if show_stamp:
            stamp_bbox_px = detect_stamp_bbox_px_from_pdf_image(
                content, page_index=page, zoom=zoom
            )

        png = render_page_with_boxes(
            content,
            layout,
            page_index=page,
            zoom=zoom,
            draw_label=labels,
            draw_value=values,
            stamp_bbox_px=stamp_bbox_px,  # 초록색 박스(서비스 레벨에서 색 지정)
        )
        return StreamingResponse(io.BytesIO(png), media_type="image/png")

    except Exception as e:
        print(">>> annotated ERROR:", e)
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/extract")
async def extract(
    file: UploadFile = File(...),
    page: int = Query(0, ge=0),
    zoom: float = Query(2.0, gt=0, le=6.0),
):
    try:
        pdf_bytes = await file.read()

        cert_code = detect_certificate_type_from_pdf(pdf_bytes)
        if cert_code == "inforproc":
            layout = extract_inforproc(pdf_bytes)
        elif cert_code == "adsp":
            layout = extract_adsp(pdf_bytes)
        elif cert_code == "sqld":
            layout = extract_sqld(pdf_bytes)
        else:
            layout = {}

        def first_val(keys):
            fields = layout.get("fields", {}) if layout else {}
            for k in keys:
                info = fields.get(k)
                if info and info.get("value_spans"):
                    return info["value_spans"][0]["text"]
            return None

        certification_name = CERT_NAME_MAP.get(cert_code)
        name = first_val(["성명"])
        certification_num = first_val(["자격번호"])
        birth_date = first_val(["생년월일"])
        admission_date = first_val(["합격 연월일", "합격일자"])
        expiration_date = first_val(["유효기간"])

        similarity = 0.0
        allowed = cert_code in ALLOWED_CERT_CODES
        stamp_bbox_px_for_upload = None

        if allowed:
            similarity, bbox_px = compute_stamp_similarity_and_bbox(
                pdf_bytes, page_index=page, zoom=zoom, cert_code=cert_code
            )
            if bbox_px and similarity > THRESHOLD:
                stamp_bbox_px_for_upload = bbox_px

        is_valid = bool(allowed and similarity > THRESHOLD)
        stamp_similarity_pct = round(similarity * 100.0, 2)

        verified_image_url = None
        if is_valid:
            annotated_png = render_page_with_boxes(
                pdf_bytes,
                layout,
                page_index=page,
                zoom=zoom,
                draw_label=True,
                draw_value=True,
                stamp_bbox_px=stamp_bbox_px_for_upload,  # 도장 박스 포함(초록색)
            )
            verified_image_url = upload_certification_image(annotated_png, ext="png")

        return JSONResponse(
            {
                "isValid": is_valid,
                "verifiedCertificationImage": verified_image_url if is_valid else None,
                "certificationName": certification_name if is_valid else None,
                "name": name if is_valid else None,
                "certificationNum": certification_num if is_valid else None,
                "birthDate": birth_date if is_valid else None,
                "admissionDate": admission_date if is_valid else None,
                "expirationdate": expiration_date if is_valid else None,
                "stampSimilarity": stamp_similarity_pct,
            },
            status_code=200,
        )

    except Exception as e:
        print(">>> extract ERROR:", e)
        print(traceback.format_exc())
        return JSONResponse(
            {
                "isValid": False,
                "verifiedCertificationImage": None,
                "certificationName": None,
                "name": None,
                "certificationNum": None,
                "birthDate": None,
                "admissionDate": None,
                "expirationdate": None,
            },
            status_code=200,
        )
