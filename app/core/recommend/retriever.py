from langchain_community.retrievers import BM25Retriever
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain.retrievers import EnsembleRetriever
from konlpy.tag import Kkma, Okt
from langchain_community.vectorstores import FAISS
from app.core.recommend.faiss import reciprocal_rank_fusion
from langchain.retrievers.multi_query import MultiQueryRetriever

def kkma_tokenize(text):
    kkma = Kkma()
    return [token for token in kkma.morphs(text)]

def okt_tokenize(text):
    okt = Okt()
    return [token for token in okt.morphs(text)]

def make_bm25_retriever(documents, tokenize = None):
    if(tokenize):
        return BM25Retriever.from_documents(documents, preprocess_func=tokenize)
    return KiwiBM25Retriever.from_documents(documents)

def ensemble_retriever(vector_retriever, bm25_retriever, llm_model):
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_retriever, llm=llm_model
    )
    return EnsembleRetriever(
        retrievers=[bm25_retriever, multi_query_retriever],
        weights=[0.2, 0.8]
    )

def rrf_result(queries, ensemble_retriever):
    results = []

    for query in queries:
        result = ensemble_retriever.invoke(query, k=3)
        results.extend(result)

    final_result = reciprocal_rank_fusion([results])
    return final_result