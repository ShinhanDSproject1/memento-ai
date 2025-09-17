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
import re

def make_document(df, chunk_size: int = 500, overlap: int = 50):
    documents = []
    for index, row in df.iterrows():
        combined_content = f"{row['mentos_title']}\n\n{row['mentos_content']}"
        chunks = sentence_chunking(combined_content, max_chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
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

# --- 청킹 함수 ---
def sentence_chunking(text: str, max_chunk_size: int = 500, overlap: int = 50):
    """
    문장 단위로 text를 나누고 max_chunk_size를 넘으면 새로운 청크로 분리
    """
    sentences = re.split(r'(?<=[.?!\n])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        if len(current_chunk) + len(sent) <= max_chunk_size:
            current_chunk += sent + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # overlap 적용
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            chunk = chunks[i]
            if i > 0:
                chunk = chunks[i-1][-overlap:] + chunk
            overlapped_chunks.append(chunk)
        return overlapped_chunks
    else:
        return chunks