import json
from app.utils.logger import setup_logger
import re
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

logger = setup_logger()

class AgentExecutionError(Exception):
    pass

class LLMAgentService:
    async def run_agent_flow(self, user_input: str, llm_client, conversation_history: list, system_message: str, financial_dict_retriever) -> tuple[str, bool]:
        """
        LLM을 호출하여 사용자의 입력에 대한 응답과 상태를 판단합니다.
        """
        rag_context = "제공된 참고 정보 없음."
        if financial_dict_retriever:
            try:
                llm_for_retriever = ChatOpenAI(
                    model_name="gpt-4o-mini",
                    max_tokens=500,
                    temperature=0,
                    api_key=llm_client.api_key
                )
                multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever=financial_dict_retriever, # 기존 리트리버를 그대로 활용
                    llm=llm_for_retriever # 질문 생성을 위한 LLM
                )
                retrieved_docs = multi_query_retriever.get_relevant_documents(query=user_input)
                if retrieved_docs:
                    # 2. 검색된 정보를 프롬프트에 넣기 좋은 형태로 가공합니다.
                    context_lines = []
                    for doc in retrieved_docs:
                        # 예시: "용어: ETF, 설명: ..., 관련 슬롯: ..."
                        # Document의 page_content나 metadata 구조에 따라 달라집니다.
                        context_lines.append(doc.page_content) 
                    rag_context = "\n---\n".join(context_lines)
            
            except Exception as e:
                logger.error(f"Glossary RAG retrieval failed: {e}", exc_info=True)
                rag_context = "참고 정보 검색 중 오류 발생."   
        CONVERSATIONAL_GUIDELINES = f"""
            [역할]
            당신은 사용자의 재정 목표 달성을 돕는 친절하고 스마트한 전문 금융 어드바이저 '토리'입니다.
            당신의 임무는 자연스러운 대화를 통해 사용자의 상황을 파악하고, 최종적으로 맞춤형 금융 정보를 추천하기 위해 필요한 핵심 정보들을 수집하는 것입니다.
            
            [수집 목표 정보 (Slots)]
            당신은 대화를 통해 아래 5가지 정보를 반드시 파악해야 합니다.
            1. 관심 분야: (주식, 부동산, 저축, 펀드, 채무 관리, 금융사기, 대출 등)
            2. 최종 목표: (내집마련, 노후준비, 목돈마련 등)
            3. 목표 금액: (구체적인 액수)
            4. 투자 경험: (초보, 경험자 등)
            5. 목표 기간: (단기, 중기, 장기 또는 구체적인 기간)
            
            [대화 전략]
            - **모든 판단은 태그로 표현합니다.** 답변 문장은 따로 생성하지 않고, 코드가 별도로 처리합니다.
            - 사용자의 답변이 기존 정보를 더 구체화하는 경우, 아래 예시와 완전히 동일한 JSON 형식으로 반환하세요.
            ✅ 올바른 예시:
            [업데이트: {{"관심 분야": "주식"}}]
            [업데이트: {{"목표 금액": "5000만원"}}]

            ❌ 잘못된 예시:
            [업데이트: {{'"관심 분야"': '주식'}}]
            [업데이트: {{"'목표 금액'": "5000만원"}}]
            
            - key 이름은 반드시 위 5개 슬롯명 중 하나여야 하며, 겹따옴표를 넣지 마세요.
            - 항상 key와 value는 큰따옴표 한 번만 사용하세요.
            
            - 모든 정보가 수집되면, **[정보완료]** 태그만 반환하세요.
            - 금융, 재테크와 무관한 질문이면 **[금융무관]** 태그만 반환하세요.
            - 그 외의 경우, 필요한 정보를 얻기 위해 2~3문장으로 자연스럽게 질문하세요.
            
            ---
            [참고 정보]
            다음은 사용자의 발언을 이해하는 데 도움이 될 수 있는 금융 용어 정보입니다. 이 정보를 바탕으로 사용자의 의도를 더 정확하게 분류하고, 해당하는 슬롯 값을 추출하세요.
            {rag_context}
            ---

            [제약 조건]
            - 답변에 [태그]가 포함된 경우, 다른 문장은 절대 추가하지 마세요.
            - JSON 태그 내 key와 value는 반드시 큰따옴표(")를 한 번만 사용해야 합니다.
            - JSON 태그 뒤에는 다른 문장을 추가하지 마세요. 태그만 반환하세요.
            - 당신의 응답은 반드시 [업데이트: {...}], [정보완료], [금융무관] 태그 중 하나이거나, 정보를 수집하기 위한 질문이어야 합니다. 이 외의 다른 형식(예: 태그 없는 JSON)의 응답은 절대 허용되지 않습니다.

            [사용자 입력]
            {user_input}

            [토리의 다음 행동 판단]
        """

        messages = [{"role": "system", "content": f"{system_message}\n{CONVERSATIONAL_GUIDELINES}"}]
        messages.extend(conversation_history)  # [{"role": "user"/"assistant", "content": "..."}]
        messages.append({"role": "user", "content": user_input})
        try:
            response_from_llm = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                top_p=0.95, 
                max_tokens=512, 
                temperature=0.2, 
                )
            llm_output = response_from_llm.choices[0].message.content.strip()
            
            # LLM의 응답에 따라 분기 처리
            if "[정보완료]" in llm_output:
                return ("final_response_needed", True)
            
            elif "[금융무관]" in llm_output:
                return ("irrelevant", False)
            
            elif "[업데이트:" in llm_output:
                try:
                    update_data_match = re.search(r'\[업데이트:\s*(\{.*?\})\]', llm_output)
                    if update_data_match:
                        raw_json = update_data_match.group(1)

                        # 🚨 여기서 key 이름 앞뒤의 따옴표 제거
                        # ex) {"'관심 분야'": "주식"}  -> {"관심 분야": "주식"}
                        cleaned_json = re.sub(r"'([가-힣\s]+)'(?=\s*:)", r'"\1"', raw_json)  # key에 작은따옴표 → 큰따옴표
                        cleaned_json = re.sub(r'"{2,}([가-힣\s]+)"{2,}(?=\s*:)', r'"\1"', cleaned_json)  # 겹따옴표 → 하나로
                        logger.info(f"파싱 전 {cleaned_json}")
                        update_data = json.loads(cleaned_json)
                        logger.info(f"파싱 후 {update_data}")
                        return (json.dumps({"update": update_data}, ensure_ascii=False), False)
                    else:
                        logger.error(f"업데이트 정규식 매칭 실패. LLM 출력: {llm_output}")
                        logger.info("업데이트 부분 오류")
                        return ("챗봇 서비스에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.", False)
                except (json.JSONDecodeError, IndexError) as e:
                    logger.error(f"업데이트 데이터 파싱 실패: {e}", exc_info=True)
                    return (llm_output, False)
            else:
                # 일반적인 대화 (추가 정보 수집 질문)
                return (llm_output, False)
        
        except Exception as e:
            logger.error(f"클라이언트 오류: {e}", exc_info=True)
            return ("챗봇 서비스에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.", False)