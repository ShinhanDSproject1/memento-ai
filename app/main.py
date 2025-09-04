from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.certificates import router as cert_router

load_dotenv()

app = FastAPI(title="Memento OCR API")

# 라우터 등록
app.include_router(cert_router, prefix="/certs", tags=["certificates"])
