from typing import List
from konlpy.tag import Okt
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# --- 의도 종합 쿼리 생성을 위한 프롬프트 ---
SYNTHESIZE_QUERY_PROMPT_TEMPLATE = """
당신은 사용자의 여러 발언을 분석하여 핵심적인 단일 의도를 찾아내는 검색 전문가입니다.
아래에 사용자가 순서대로 입력한 문장들이 있습니다.
당신은 금융,재테크 전문가로서 이 문장들을 모두 고려하여, 사용자의 근본적인 목표를 가장 잘 나타내는 하나의 완벽한 질문으로 재작성해주세요.
이 질문은 벡터 데이터베이스에서 관련 문서를 검색하는 데 사용될 것입니다.
**[중요 규칙]**
**- 원본에 포함된 구체적인 금액, 숫자, 고유명사 등 핵심 정보는 절대 변경하거나 생략해서는 안 됩니다.**
**- 원본의 모든 핵심 의도를 충실히 반영해야 합니다.**
**- 관심사(예: 투자, 저축등)이 문장 제일 앞에 오도록 합니다.또한 관심사는 자연스러운 문장으로 구성합니다.(예: '투자:' 이런거 금지).**
다른 설명 없이, 오직 최종 질문 하나만 생성해주세요.

[사용자 발언 목록]
{original_queries}
"""
SYNTHESIZE_QUERY_PROMPT = PromptTemplate.from_template(SYNTHESIZE_QUERY_PROMPT_TEMPLATE)

# --- LLM 호출 래퍼 함수 ---
def invoke_gradio_llm_re(input_data, llm_client):
    prompt_text = input_data.to_string() if hasattr(input_data, 'to_string') else str(input_data)
    result = llm_client.predict(
        prompt_text,
        top_p=0.95, 
        max_tokens=1024,
        temperature=0.25,
        api_name="/chat"
    )
    return str(result[0]) if isinstance(result, (list, tuple)) and result else str(result)

# --- 의도 종합 쿼리 생성 함수 ---
def synthesize_query(queries: List[str], llm_client):
    print(f"Synthesizing intent from: {queries}")
    # 여러 쿼리를 줄바꿈으로 연결하여 프롬프트에 삽입
    queries_str = "\n- ".join(queries)
    prompt = SYNTHESIZE_QUERY_PROMPT.format(original_queries=queries_str)
    
    # LLM을 호출하여 새로운 쿼리 생성
    synthesized_query = invoke_gradio_llm_re(prompt, llm_client)
    print(f"Synthesized query: {synthesized_query}")
    return synthesized_query

# --- 최종 앙상블 리트리버 빌더 ---
def build_ensemble_retrieve_re(vector_retriever, bm25_retriever):
    # (의도 종합 쿼리가 MultiQuery의 역할을 대신함)
    
    # 두 리트리버를 RRF로 융합
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.25, 0.75] # BM25와 벡터 검색 가중치 조절
    )