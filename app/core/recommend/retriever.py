from langchain_community.retrievers import BM25Retriever
from konlpy.tag import Kkma, Okt

def kkma_tokenize(text):
    kkma = Kkma()
    return [token for token in kkma.morphs(text)]

def okt_tokenize(text):
    okt = Okt()
    return [token for token in okt.morphs(text)]

#내부 코드에 k=4가 명시되어 있음
def make_bm25_retriever(documents, tokenize=None):
    return BM25Retriever.from_documents(documents, preprocess_func=tokenize)