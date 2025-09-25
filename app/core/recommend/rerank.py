from typing import List
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from app.utils.logger import setup_logger
from konlpy.tag import Kkma
from langchain_core.documents import Document
import re

kkma = Kkma()

# 불용어 리스트
STOPWORDS = {
    "이", "가", "은", "는", "을", "를", "에", "의", "와", "과", "도", "으로", "부터", "까지",
    "있다", "있습니다", "합니다", "했다", "되다", "되요", "입니다", "하다", "해요",
    "것", "거", "더", "같다", "나", "우리", "그리고", "하지만", "또한", "또", "등", "만",
    "오늘", "이번", "이번에", "최근", "매일", "자주", "언제", "어제", "내일", "항상",
    "있어", "있어요", "싶다", "좋다", "좋아요", "많다", "많아요", "중", "중에"
}
logger = setup_logger()
# --- 의도 종합 쿼리 생성을 위한 프롬프트 ---
SYNTHESIZE_QUERY_PROMPT_TEMPLATE = """
당신은 사용자의 여러 발언을 분석하여 핵심적인 단일 의도를 찾아내는 검색 전문가입니다.
아래에 사용자가 순서대로 입력한 문장들이 있습니다.
당신은 금융,재테크 전문가로서 이 문장들을 모두 고려하여, 사용자의 근본적인 목표를 가장 잘 나타내는 하나의 완벽한 질문으로 재작성해주세요.
이 질문은 벡터 데이터베이스에서 관련 문서를 검색하는 데 사용될 것입니다.

**중요 규칙**
- 원본에 포함된 구체적인 금액, 숫자, 고유명사 등 핵심 정보는 절대 변경하거나 생략하지 마세요.
- 원본의 모든 핵심 의도를 반영하세요.
- 관심사(예: 투자, 저축 등)가 문장 제일 앞에 오도록 자연스럽게 구성하세요.

[사용자 발언 목록]
{original_queries}
"""
SYNTHESIZE_QUERY_PROMPT = PromptTemplate.from_template(SYNTHESIZE_QUERY_PROMPT_TEMPLATE)

# --- LLM 호출 래퍼 함수 ---
def invoke_llm(prompt_text, llm_client, system_message=""):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt_text})

    result = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1024,
        temperature=0.25,
        top_p=0.95,
    )
    return result.choices[0].message.content.strip()

# --- 의도 종합 쿼리 생성 함수 ---
def synthesize_query(queries: List[str], llm_client, system_message=""):
    logger.info(f"Synthesizing intent from: {queries}")
    queries_str = "\n- ".join(queries)
    prompt = SYNTHESIZE_QUERY_PROMPT.format(original_queries=queries_str)
    synthesized_query = invoke_llm(prompt, llm_client, system_message)
    logger.info(f"Synthesized query: {synthesized_query}")
    return synthesized_query

# --- 최종 앙상블 리트리버 빌더 ---
def build_ensemble_retrieve_re(vector_retriever, bm25_retriever):
    
    #EnsembleRetriever가 RRF(Reciprocal rank fusion) 내장
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.25, 0.75]
    )

def extract_keywords_kkma(queries: List[str]) -> List[str]:
    keywords = set()
    for q in queries:
        nouns = kkma.nouns(q)
        for n in nouns:
            if n not in STOPWORDS and len(n) > 1:
                keywords.add(n)
    return list(keywords)

def rerank_documents_with_keywords(documents: list[Document], keywords: list[str], keyword_weight: float = 0.1):
    keywords_set = set(keywords)
    reranked_docs = []

    for doc in documents:
        keyword_count = sum(1 for kw in keywords_set if kw in doc.page_content)
        adjusted_score = doc.metadata.get('score', 0) * (1 + keyword_weight * keyword_count / max(len(keywords_set), 1))
        new_metadata = doc.metadata.copy()
        new_metadata['adjusted_score'] = adjusted_score
        reranked_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))

    return reranked_docs