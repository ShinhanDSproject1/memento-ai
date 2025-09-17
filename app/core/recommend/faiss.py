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