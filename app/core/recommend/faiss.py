import pandas as pd
from sentence_transformers.readers import InputExample
from langchain_core.documents import Document
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.load import dumps, loads
from typing import List
from sqlalchemy.orm import Session
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import text 

def make_document(df):
    documents = []
    for index, row in df.iterrows():
        doc = Document(
            page_content=row['mentos_content'],
            title = row['mentos_title'],
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

def reciprocal_rank_fusion(results: List[List[Document]], k=60):
    fused_scores = {}
    doc_map = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_key = doc.page_content
            
            if doc_key not in fused_scores:
                fused_scores[doc_key] = 0
                doc_map[doc_key] = doc
            
            fused_scores[doc_key] += 1 / (rank + 1 + k)

    reciprocal_rank_fusion_results = [
        (doc_map[doc_key], score)
        for doc_key, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reciprocal_rank_fusion_results

def make_vector_retriever(vectorstore):
    return vectorstore.as_retriever()

def select_mentos_data_to_df(db: Session):
    result = db.execute(text("SELECT mentos_seq, mentos_title, mentos_content FROM mentos WHERE status = 'ACTIVE'")).fetchall()
    mentos_df = pd.DataFrame(result, columns=['mentos_seq', 'mentos_title', 'mentos_content'])

    mentos_df['mentos_content'] = mentos_df['mentos_content'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text(strip=True))

    return mentos_df