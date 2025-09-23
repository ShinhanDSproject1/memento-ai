from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from konlpy.tag import Kkma, Okt
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

QUERY_PROMPT_TEMPLATE = """
당신은 AI 언어 모델 어시스턴트이며 금융, 재테크 전문가입니다.
당신의 임무는 주어진 질문을 바탕으로, 그 질문에 대한 답을 찾기 위해 벡터 데이터베이스에 사용할 만한
3개의 다른 버전의 질문을 생성하는 것입니다.
질문의 의미는 유지하되, 사용자들이 검색할 만한 다양한 관점의 표현으로 바꿔주세요.
오직 질문 목록만 반환하고, 다른 설명은 덧붙이지 마세요.

원본 질문: {question}
"""

# PromptTemplate 객체로 생성
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=QUERY_PROMPT_TEMPLATE,
)

#무겁고 score는 비슷하나 의도와 전혀 다른 추천 리스트
def kkma_tokenize(text):
    kkma = Kkma()
    return [token for token in kkma.morphs(text)]

def okt_tokenize(text):
    okt = Okt()
    return [token for token in okt.morphs(text)]

#내부 코드에 k=4가 명시되어 있음
def make_bm25_retriever(documents, tokenize=None):
    return BM25Retriever.from_documents(documents, preprocess_func=tokenize)

def invoke_gradio_llm(input_data, llm_model):
    """
    프롬프트에서 온 입력을 받아 Gradio 엔드포인트를 호출하고,
    그 결과를 반환하는 함수입니다.
    """
    # 프롬프트의 결과물(PromptValue)을 문자열로 변환합니다.
    prompt_text = input_data.to_string()

    result = llm_model.predict(
        prompt_text,
        top_p=0.95, 
        max_tokens=1024, 
        temperature=0.3,
        api_name="/chat"
    )
    # Gradio의 출력값에서 실제 텍스트 응답을 추출합니다.
    # 이 부분은 사용하는 Gradio 앱의 반환 형식에 따라 조정이 필요합니다.
    if isinstance(result, str):
        return result
    elif isinstance(result, (list, tuple)) and len(result) > 0:
        return str(result[0])
    else:
        # 예상치 못한 결과 형식에 대한 기본 처리
        return str(result)

def ensemble_retriever(vector_retriever, bm25_retriever, llm_model):
    llm_as_runnable = RunnableLambda(
        lambda input_data: invoke_gradio_llm(input_data, llm_model)
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_retriever,
        llm=llm_as_runnable,
        prompt=QUERY_PROMPT,
        k=50
    )
    #EnsembleRetriever가 RRF(Reciprocal rank fusion) 내장
    return EnsembleRetriever(
        retrievers=[bm25_retriever, multi_query_retriever],
        weights=[0.2, 0.8]
    )