from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.certificates import router as cert_router
from app.api.ai import router as ai_router
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import mysql.connector
from app.db.database import get_db
import os
from gradio_client import Client
from app.utils.logger import setup_logger

load_dotenv()
logger = setup_logger

# lifespan 이벤트 핸들러
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    try:
        get_db()
        logger.info("DB connect success!")
    except mysql.connector.Error as err:
        logger.error("DB connect error: %s", err, exc_info=True)
    
    try:
        model_name = "josangho99/ko-paraphrase-multilingual-MiniLM-L12-v2-multiTask"
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
        app.state.system_message = "You are a helpful assistant. And you must speak Korean. And you financial professional"
        logger.info("LLM model loading success!")
    except Exception as e:
        logger.error("LLM model loading error: %s", e, exc_info=True)
        app.state.llm = None

    yield
    logger.info("Application shutdown...")
    app.state.llm = None
    app.state.model = None
        

app = FastAPI(title="Memento OCR API", lifespan=lifespan)

# 라우터 등록
app.include_router(cert_router, prefix="/certs", tags=["certificates"])
app.include_router(ai_router, prefix="/ai", tags=["ai"])