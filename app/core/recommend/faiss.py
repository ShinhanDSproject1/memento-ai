import pandas as pd
from sentence_transformers.readers import InputExample
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.load import dumps, loads
from sqlalchemy.orm import Session
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import text
from langchain.document_loaders import PyPDFLoader
from app.utils.logger import setup_logger
from app.schemas.recommend_mentos import RecommendData
import re

logger = setup_logger()

def make_document(df):
    documents = []
    for index, row in df.iterrows():
        # Combine title and content into a single string
        combined_content = f"{row['mentos_title']}\n\n{row['mentos_content']}"
        
        # Create a single Document for the entire combined content
        doc = Document(
            page_content=combined_content,
            metadata={'mentos_seq': row['mentos_seq'], 'title': row['mentos_title']}
        )
        documents.append(doc)
    return documents

def vector_embedding(documents, embeddings_model):
    vectorstore = FAISS.from_documents(
        documents,
        embedding = embeddings_model,
        distance_strategy = DistanceStrategy.COSINE
    )
    return vectorstore

def make_vector_retriever(vectorstore):
    return vectorstore.as_retriever()

def select_mentos_data_to_df(db: Session):
    result = db.execute(text("SELECT mentos_seq, mentos_title, mentos_content FROM mentos WHERE status = 'ACTIVE'")).fetchall()
    mentos_df = pd.DataFrame(result, columns=['mentos_seq', 'mentos_title', 'mentos_content'])

    mentos_df['mentos_content'] = mentos_df['mentos_content'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text(strip=True))

    return mentos_df

def select_recommend_data_by_mentos_seq(db:Session, mentos_seq_list):
    result = db.execute(text("""SELECT m.mentos_seq, m.mentos_image, m.mentos_title, m.price, mp.mento_profile_image 
                            FROM mentos m
                            LEFT JOIN member ON m.member_seq = member.member_seq
                            LEFT JOIN mento_profile mp ON member.member_seq = mp.member_seq
                            WHERE mentos_seq IN :mentos_seq"""), 
                        {"mentos_seq": tuple(mentos_seq_list)})
    rows = result.fetchall()
    logger.info(rows)
    return [RecommendData(
        mentos_seq=row[0],
        mentos_image=row[1],
        mentos_title=row[2],
        price=row[3],
        mento_profile_image=row[4]
    ) for row in rows]

def financial_dict_pdf_load():
    loader = PyPDFLoader("/assets/2020_경제금융용어 700선_게시.pdf")
    texts = loader.load_and_split()
    texts = texts[13:]
    texts = texts[:-1]
    return texts