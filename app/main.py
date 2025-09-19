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
from openai import OpenAI
from app.core.recommend.faiss import(
    make_document,
    vector_embedding,
    make_vector_retriever,
    select_mentos_data_to_df,
    financial_dict_pdf_load,
)
from app.utils.utils import (
    find_boilerplate_phrases, 
    remove_phrases_from_text,
    )
from app.core.recommend.retriever import(
    okt_tokenize,
    make_bm25_retriever,
)
from app.core.recommend.rerank import(
    build_ensemble_retrieve_re,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import threading

load_dotenv()
logger = setup_logger()
app_state_lock = threading.Lock()
scheduler = AsyncIOScheduler()

def setup_search_retrievers(app: FastAPI):
    """인덱스를 새로 빌드하는 함수"""
    logger.info("Starting scheduled index refresh...")
    try:
        db = next(get_db())
        model = app.state.model
        mentos_df = select_mentos_data_to_df(db)
        boilerplate_list = find_boilerplate_phrases(mentos_df, content_column='mentos_content')
        mentos_df['mentos_content'] = mentos_df['mentos_content'].apply(
            lambda text: remove_phrases_from_text(text, boilerplate_list)
        )
        mentos_documents = make_document(mentos_df)
        mentos_vectorstore = vector_embedding(mentos_documents, model)
        mentos_vector_retriever = make_vector_retriever(mentos_vectorstore)
        Otk_retriever = make_bm25_retriever(mentos_documents, okt_tokenize)
        db.close()
        ensemble_retriever =  build_ensemble_retrieve_re(mentos_vector_retriever, Otk_retriever)

        with app_state_lock:
            app.state.ensemble_retriever = ensemble_retriever

        logger.info("Scheduled index refresh complete.")
        db.close()
    except Exception as e:
        logger.error(f"Scheduled index refresh failed: {e}", exc_info=True)
    finally:
        db.close()

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
        db.close()
    except mysql.connector.Error as err:
        logger.error("DB connect error: %s", err, exc_info=True)
    
    try:
        #model_name = "josangho99/ko-paraphrase-multilingual-MiniLM-L12-v2-multiTask"
        #model_name = "korruz/bge-base-financial-matryoshka"
        #model_name = "intfloat/multilingual-e5-small"
        model_name = "intfloat/multilingual-e5-base"
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
        API_KEY = os.getenv("OPENAI_API_KEY")
        app.state.llm = OpenAI(api_key=API_KEY)
        app.state.system_message = "당신은 한국어를 사용하는 친절한 금융 전문가입니다. 재테크와 금융에 대한 질문에 답변해 주세요."
        logger.info("LLM model loading success!")
    except Exception as e:
        logger.error("LLM model loading error: %s", e, exc_info=True)
        app.state.llm = None
    
    try:
        setup_search_retrievers(app)
        #scheduler.add_job(setup_search_retrievers, 'cron', hour=4, args=[app]) #오전 4시...
        #2시간마다 멘토스 벡터스토어 re-indexing
        scheduler.add_job(setup_search_retrievers, trigger='interval', hours=2, args=[app])
        scheduler.start()
        logger.info("Retriever refresh")
    except Exception as e:
        logger.error("Mentos vectorstore not loading...")
        app.state.ensemble_retriever = None

    try:
        reranker_model_url = 'kkresearch/bge-reranker-v2-m3-korean-finance'
        app.state.reranker = CrossEncoder(reranker_model_url)

    except Exception as e:
        logger.error("Reranker model loading error: %s", e, exc_info=True)
        app.state.reranker = None
    try:
        financial_dict_documents = financial_dict_pdf_load()
        financial_dict_vectorstore = vector_embedding(documents = financial_dict_documents, embeddings_model = app.state.model)
        app.state.financial_dict_retriever = make_vector_retriever(financial_dict_vectorstore)
    except Exception as e:
        logger.error("Not Found Financial dictionary...")
        app.state.financial_dict_retriever = None
    app.state.conversation_memory = {}
    yield
    logger.info("Application shutdown...")
    tunnel.close()
    app.state.llm = None
    app.state.model = None
    app.state.ensemble_retriever=None
    app.state.financial_dict_retriever = None
    app.state.reranker = None

app = FastAPI(title="Memento API", lifespan=lifespan)

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