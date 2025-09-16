import json
from app.utils.tools import save_user_data, check_profile_complete, save_meaningful_data
from app.utils.logger import setup_logger
import re

logger = setup_logger()

class AgentExecutionError(Exception):
    pass

# Helper: Generates the final, ready-for-recommendation response.
async def generate_final_response(llm_client, system_message, conversation_history, user_input, tool_result):
    final_response_prompt = f"""
    {system_message}
    {conversation_history}
    사용자: {user_input}
    도구 실행 결과: {tool_result}

    지시사항:
    - 모든 정보가 성공적으로 수집되었습니다. 이제 사용자에게 맞춤형 강의를 추천할 준비가 되었음을 알리고, 추천을 받을지 확인하는 문장을 생성하세요.
    - 예: "모든 정보가 수집되었습니다. 맞춤형 강의를 추천받으시겠어요?"와 같이 구체적으로 물어보세요.
    - 다른 설명은 최소화하고, 확인 메시지 자체에 집중하세요.
    - 직접 강의를 추천하지 말아주세요.
    최종 응답:
    """
    final_response = llm_client.predict(final_response_prompt, top_p=0.95, max_tokens=1024, temperature=0.7, api_name="/chat")
    return final_response.strip()

# Helper: Generates a follow-up question when more info is needed.
async def generate_followup_question(llm_client, system_message, conversation_history, user_input, tool_result):
    followup_prompt = f"""
    {system_message}
    {conversation_history}
    사용자: {user_input}
    도구 실행 결과: {tool_result}

    지시사항:
    - 직접 강의를 추천하지 말아주세요.
    - 현재 대화의 목표는 사용자로부터 관심사, 수준, 목표를 모두 수집하는 것입니다.
    - 현재 수집된 정보가 불완전합니다.
    - **먼저 사용자의 관심사에 대한 흥미로운 정보를 간결하게 하나 제공하고, 이어서 부족한 정보를 요청하는 질문을 단 하나만 생성하세요.**
    - 예시: "주식에 투자하면 기업의 성과에 따라 배당금을 받거나 주가가 오르는 이익을 얻을 수 있어요. 혹시 투자 경험이 어느 정도 되시나요?"
    최종 응답:
    """
    response = llm_client.predict(followup_prompt, top_p=0.95, max_tokens=1024, temperature=0.7, api_name="/chat")
    return response.strip()


class LLMAgentService:
    async def run_agent_flow(self, user_id: str, user_input: str, llm_client, conversation_history: str, system_message: str) -> tuple[str, bool]:
        TOOLS_LIST_FOR_LLM = """
            이용 가능한 도구 목록:
            1. save_user_data(data: dict): 사용자의 관심사, 수준, 목표를 JSON 형식으로 저장합니다. **사용자가 금액을 목표로 제시할 경우, 해당 금액을 'goal' 필드에 저장합니다.** 예시: '1억 모으기'라는 수치적인 목표 입력이 있다면 save_user_data(data={'interest': '주식', 'level': '초보', 'goal': '1억 모으기'})를 호출합니다. 명확한 목표가 없다면 'goal' 필드를 None으로 둡니다.
            2. check_profile_complete(): 필요한 모든 정보(관심사, 수준, 목표)가 수집되었는지 확인합니다. 이 도구는 `True` 또는 `False`를 반환합니다.
            3. save_meaningful_data(text: str): 사용자의 자연스러운 관심사나 질문을 발견했을 때 해당 문장을 저장합니다.

            도구 사용 규칙:
            - 반드시 'Tool: [도구명]' 형태로 도구를 호출해야 합니다.
            - 'Tool:' 뒤에 한 칸 띄우고 도구명과 매개변수를 JSON 형태로 작성해야 합니다.
            """
        
        CONVERSATIONAL_GUIDELINES = """
        ## 지침 및 페르소나
        당신은 한국어 금융 전문가 AI입니다. 다음 규칙을 따르세요:
        - 사용자의 목표: 친절하고 유연하게 대화하며, 사용자의 질문에 답하고 필요한 경우 정보를 요청하세요.
        - 직접 강의를 추천하지 말아주세요.
        - 답변 금지: 금융과 무관한 질문에는 답변하지 마세요. 대신, "죄송하지만, 저는 금융과 재테크 분야에 대한 질문에만 답변할 수 있어요. 혹시 이와 관련하여 궁금한 점이 있으신가요?"와 같이 대화를 재개하세요.
        - **도구 호출은 명확하게 필요한 경우에만 하세요. 그 외 모든 상황에서는 자연어로 답변하세요.**
        """

        prompt = f"""
            {system_message}
            {CONVERSATIONAL_GUIDELINES}
            {TOOLS_LIST_FOR_LLM}
            대화 기록: {conversation_history}
            사용자: {user_input}
            챗봇의 다음 행동:
            """
        try:
            response_from_llm = llm_client.predict(prompt, top_p=0.95, max_tokens=1024, temperature=0.7, api_name="/chat")
            llm_output = response_from_llm.strip()
            logger.info(f"LLM Output: {llm_output}")
            
            final_response_text = ""
            is_complete_bool = False

            # LLM이 도구 호출을 시도했는지 확인
            if "Tool:" in llm_output:
                try:
                    # 불필요한 태그 및 텍스트 제거
                    cleaned_output = re.sub(r'<\|.*?\|>', '', llm_output).strip()
                    cleaned_output = re.sub(r'We have called tool. Now respond.', '', cleaned_output).strip()
                    
                    tool_call_str = cleaned_output[cleaned_output.find("Tool:"):].strip()
                    json_match = re.search(r'\{.*?\}', tool_call_str)
                    
                    # JSON 객체가 없는 경우, 도구 호출이 아닌 것으로 간주하고 자연어 답변으로 처리
                    if not json_match:
                        final_response_text = cleaned_output.replace("Tool:", "").strip()
                        is_complete_bool = False
                    else:
                        valid_json_str = json_match.group(0).replace("'", '"').replace("None", "null")
                        tool_call = json.loads(valid_json_str)

                        tool_name = tool_call.get("tool_name")
                        tool_args = tool_call.get("tool_args", {})

                        if tool_name == "save_user_data":
                            save_user_data(user_id=user_id, data=tool_args)
                            is_complete_check = check_profile_complete(user_id=user_id)
                        
                            if is_complete_check == "True":
                                final_response_text = await generate_final_response(llm_client, system_message, conversation_history, user_input, "사용자 프로필 완성됨.")
                                is_complete_bool = True
                            else:
                                final_response_text = await generate_followup_question(llm_client, system_message, conversation_history, user_input, "프로필이 아직 완성되지 않았음.")
                                is_complete_bool = False
                        
                        elif tool_name == "check_profile_complete":
                            is_complete_check = check_profile_complete(user_id=user_id)
                            if is_complete_check == "True":
                                final_response_text = await generate_final_response(llm_client, system_message, conversation_history, user_input, "사용자 프로필 완성됨.")
                                is_complete_bool = True
                            else:
                                final_response_text = await generate_followup_question(llm_client, system_message, conversation_history, user_input, "프로필이 아직 완성되지 않았음.")
                                is_complete_bool = False
                        
                        else:    
                            # 알 수 없는 도구 호출 시 LLM의 원래 응답으로 대체
                            final_response_text = cleaned_output.replace("Tool:", "").strip()
                            is_complete_bool = False
                
                except (json.JSONDecodeError, IndexError) as e:
                    logger.error(f"Failed to parse tool call: {llm_output}", exc_info=True)
                    # 파싱 실패 시 자연어 답변으로 대체
                    final_response_text = re.sub(r'<\|.*?\|>', '', llm_output).strip()
                    is_complete_bool = False
            else:
                final_response_text = re.sub(r'<\|.*?\|>', '', llm_output).strip()
                is_complete_bool = False

            return (final_response_text, is_complete_bool)
        
        except Exception as e:
            logger.error(f"챗봇 서비스에 문제가 발생했습니다: {e}", exc_info=True)
            return ("챗봇 서비스에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.", False)