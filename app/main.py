from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.certificates import router as cert_router
from app.api.ai import router as ai_router
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import mysql.connector
from app.db.database import get_db
import os
from gradio_client import Client
from app.utils.logger import setup_logger
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.db.database import tunnel

load_dotenv()
logger = setup_logger()

# lifespan 이벤트 핸들러
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    try:
        tunnel.start()
        db = next(get_db())
        result = db.execute(text("SELECT 1"))
        value = result.fetchone()[0]
        logger.info(f"DB connection success! Query result: {value}")
    except mysql.connector.Error as err:
        logger.error("DB connect error: %s", err, exc_info=True)
    
    try:
        #model_name = "josangho99/ko-paraphrase-multilingual-MiniLM-L12-v2-multiTask"
        model_name = "intfloat/multilingual-e5-small"
        app.state.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
        )
        logger.info("AI model loading success!")
    except Exception as e:
        logger.error("AI model loading error: %s", e, exc_info=True)
        app.state.model = None

    try:
        API_URL = "josangho99/memento-chatbot"
        app.state.llm = Client(API_URL)
        app.state.system_message = "당신은 한국어를 사용하는 친절한 금융 전문가입니다. 재테크와 금융에 대한 질문에 답변해 주세요."
        logger.info("LLM model loading success!")
    except Exception as e:
        logger.error("LLM model loading error: %s", e, exc_info=True)
        app.state.llm = None

    try:
        reranker_model_url = 'kkresearch/bge-reranker-v2-m3-korean-finance'
        #reranker_model_url = 'Dongjin-kr/ko-reranker'
        app.state.reranker = CrossEncoder(reranker_model_url)

    except Exception as e:
        logger.error("Reranker model loading error: %s", e, exc_info=True)
        app.state.reranker = None

    yield
    logger.info("Application shutdown...")
    tunnel.close()
    app.state.llm = None
    app.state.model = None

app = FastAPI(title="Memento OCR API", lifespan=lifespan)

origins = [
    "http://memento.shinhanacademy.co.kr",
    "https://memento.shinhanacademy.co.kr",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:9999",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(cert_router, prefix="/certs", tags=["certificates"])
app.include_router(ai_router, prefix="/ai", tags=["ai"])