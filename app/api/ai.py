from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from app.schemas.text_schema import RecommendationRequest
from app.schemas.message import Message
from app.db.database import get_db
import pandas as pd
import json
import re
from fastapi.responses import JSONResponse
from app.utils.logger import setup_logger
from app.utils.utils import deduplicate_results
from app.core.recommend.chat_bot import LLMAgentService

from app.core.recommend.faiss import(
    make_document,
    vector_embedding,
    make_vector_retriever,
    select_mentos_data_to_df
)
from app.core.recommend.retriever import(
    okt_tokenize,
    make_bm25_retriever,
    ensemble_retriever,
)

from app.core.recommend.rerank import(
    extract_keywords_kkma,
    rerank_documents_with_keywords,
    synthesize_query,
    build_ensemble_retrieve_re,
)

logger = setup_logger()
router = APIRouter()
llm_service = LLMAgentService()
    
@router.get("/chatbot/{member_seq}")
async def chatbot_by_gpt(request: Request):
    llm_model = request.app.state.llm

    if not llm_model:
        return JSONResponse(
                {"error": "LLM 모델이 로드되지 않았습니다."},
                status_code=500,
        )
    
    result = '안녕하세요! 당신의 똑똑한 금융 친구 토리예요. 저와의 대화를 통해 복잡한 금융 이야기를 재미있게 풀어가고, 당신에게 꼭 맞는 서비스를 찾아 드릴게요. 지금 혹시 가장 고민하고 있거나 궁금한 점이 있으신가요? 저와의 대화를 통해 복잡한 금융 이야기를 재미있게 풀어가고, 당신에게 꼭 맞는 서비스를 찾아 드릴게요. 지금 혹시 가장 고민하고 있거나 궁금한 점이 있으신가요?'
    
    return JSONResponse(
        {
            "message": result,
        },
        status_code=200,
    )

@router.post("/chatbot")
async def chat_with_bot(request: Request, message: Message):
    llm_model = request.app.state.llm
    system_message = request.app.state.system_message
    conversation_memory = request.app.state.conversation_memory

    if not llm_model:
        return JSONResponse({"error": "LLM 모델이 로드되지 않았습니다."}, status_code=500)

    member_seq = message.member_seq
    user_input = message.content
    logger.info(f"사용자 {member_seq}의 새 입력: {user_input}")

    history_list = conversation_memory.get(member_seq, [])
    messages = [{"role": "system", "content": system_message}]
    
    for item in history_list:
        messages.append({
            "role": item["role"],
            "content": item["content"]
        })
    messages.append({"role": "user", "content": user_input})

    try:
        llm_agent_response, is_ready_check = await llm_service.run_agent_flow(
            user_input=user_input,
            llm_client=llm_model,
            conversation_history=history_list,
            system_message=system_message
        )
        logger.info(f"LLM 에이전트 응답: {llm_agent_response}")
        logger.info(f"추천 준비 상태: {is_ready_check}")
        
        final_response_text = ""
        recommendation_data = None

        if llm_agent_response == "irrelevant":
            final_response_text = (
                "죄송하지만, 저는 금융과 재테크 분야에 대한 질문에만 답변할 수 있어요."
                " 혹시 이와 관련하여 궁금한 점이 있으신가요?"
            )
            history_list.append({"role": "user", "content": user_input, "topic": "general"})
            history_list.append({"role": "assistant", "content": final_response_text, "topic": "general"})
            
            logger.info("무관한 질문으로 판단되어 일반 대화로 기록합니다.")
        elif llm_agent_response.startswith('{"update":'):
            try:
                # 이 부분을 별도의 try-except 블록으로 감싸야 함
                update_data_json_string = re.search(r'\{.*?\}', llm_agent_response).group(0)
                update_data = json.loads(update_data_json_string)
                logger.info(f"성공적으로 파싱된 업데이트 데이터: {update_data}")
                # 이제 update_data 딕셔너리를 사용
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"JSON 파싱 오류! LLM 응답: {llm_agent_response}", exc_info=True)
                final_response_text = "죄송하지만, 응답을 처리하는 중 오류가 발생했어요. 다시 말씀해주시겠어요?"
                history_list.append({"role": "user", "content": user_input, "topic": "general"})
                history_list.append({"role": "assistant", "content": final_response_text, "topic": "general"})
                conversation_memory[member_seq] = history_list
                return JSONResponse(
                    {"response": final_response_text, "recommendation_ready": is_ready_check}, status_code=500
                )
            
            is_updated = False
            for item in history_list:
                # 사용자의 발화만 업데이트
                if item.get('role') == 'user' and item.get('topic') == list(update_data.keys())[0]:
                    logger.info(f"기존 항목 업데이트: {item['content']} -> {user_input}")
                    item['content'] = user_input
                    is_updated = True
                    break
            
            if not is_updated:
                history_list.append({"role": "user", "content": user_input, "topic": list(update_data.keys())[0]})
                logger.info(f"새로운 항목 추가: {user_input} (주제: {list(update_data.keys())[0]})")
            
            second_messages = [{"role": "system", "content": system_message}]
            for item in history_list:
                second_messages.append({"role": item["role"], "content": item["content"]})
                second_messages.append({"role": "user", "content": user_input})
            
            second_response = llm_model.chat.completions.create(
                model="gpt-4o-mini",
                messages=second_messages,
                max_tokens=1024,
                temperature=0.2,
                top_p=0.95
            )
            
            final_response_text = second_response.choices[0].message.content.strip()
            history_list.append({"role": "assistant", "content": final_response_text, "topic": "general"})
            logger.info("정보 업데이트 완료 및 2차 LLM 응답 기록")

        elif llm_agent_response == "final_response_needed":
            logger.info("정보 수집 완료! 최종 응답 생성 시작.")
            final_response_text = "이제 맞춤 정보를 찾아드릴 준비가 된 것 같아요! 추천해 드릴까요?"
            # 최종 대화도 기록
            history_list.append({"role": "user", "content": user_input, "topic": "general"})
            history_list.append({"role": "assistant", "content": final_response_text, "topic": "general"})

            meaningful_sentences = [
                item['content'] for item in history_list
                if item['role'] == 'user' and item.get('topic') and item.get('topic') != 'general'
            ]
            recommendation_data = meaningful_sentences
            logger.info(f"추천 데이터: {recommendation_data}")
            
        else:
            final_response_text = llm_agent_response
            history_list.append({"role": "user", "content": user_input, "topic": "general"})
            history_list.append({"role": "assistant", "content": final_response_text, "topic": "general"})
            logger.info("일반적인 대화 응답 기록")

        conversation_memory[member_seq] = history_list

        response_payload = {
            "response": final_response_text,
            "recommendation_ready": is_ready_check,
        }
        
        if recommendation_data:
            response_payload["recommendation_data"] = recommendation_data

        if is_ready_check:
        # ✅ 중복 제거 + 오류 응답 제거
            cleaned_history = []
            seen_pairs = set()

            for item in history_list:
                # "챗봇 서비스에 문제가 발생했습니다." 같은 오류 메시지는 제외
                if item["content"].startswith("챗봇 서비스에 문제가 발생했습니다"):
                    continue
                
                # 같은 user content + topic 조합은 한 번만 저장
                key = (item["role"], item["content"], item.get("topic"))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    cleaned_history.append(item)
            
            user_history = [item["content"] for item in cleaned_history if item["role"] == "user"]

            response_payload["conversation_history"] = user_history
        logger.info(f"최종 응답 페이로드: {response_payload}")        
        return JSONResponse(response_payload, status_code=200)

    except Exception as e:
        logger.error("알 수 없는 예외 발생", exc_info=True)
        
        # ✅ 대화는 유지하되, 에러 상태를 안내
        error_message = "죄송하지만 지금은 답변을 할 수 없어요. 잠시 후 다시 시도해 주시겠어요?"
        history_list.append({"role": "user", "content": user_input, "topic": "general"})
        history_list.append({"role": "assistant", "content": error_message, "topic": "general"})
        conversation_memory[member_seq] = history_list

        return JSONResponse({
            "response": error_message,
            "recommendation_ready": False,
            "error": True
        }, status_code=200)


#종합 쿼리(각 문장을 하나로 통합한 문장)로 invoke후 re-rank
@router.post("/recommend")
async def recommend(request: Request, data: RecommendationRequest, db: Session = Depends(get_db)):
    model = request.app.state.model
    llm_model = request.app.state.llm
    reranker_model = request.app.state.reranker
    system_message = request.app.state.system_message

    if not model:
        return JSONResponse({"result": "임베딩 모델이 로드되지 않았습니다."}, status_code=500)
    if not llm_model:
        return JSONResponse({"result": "LLM 모델이 로드되지 않았습니다."}, status_code=500)
    if not reranker_model:
        return JSONResponse({"result": "Reranker 모델이 로드되지 않았습니다."}, status_code=500)
    
    member_seq = data.member_seq
    queries_list = data.queries

    mentos_df = select_mentos_data_to_df(db)
    logger.info(f"DataFrame loaded. Shape: {mentos_df.shape}")
    
    mentos_documents = make_document(mentos_df)
    logger.info("Documents created successfully.")

    mentos_vectorstore = vector_embedding(mentos_documents, model)
    mentos_vector_retriever = make_vector_retriever(mentos_vectorstore)
    Otk_retriever = make_bm25_retriever(mentos_documents, okt_tokenize)
    mentos_ensemble_retriever = build_ensemble_retrieve_re(mentos_vector_retriever, Otk_retriever)
    
    synthesized_query = synthesize_query(queries_list, llm_model, system_message)
    keywords = extract_keywords_kkma(queries_list)

    retrieved_docs = mentos_ensemble_retriever.invoke(synthesized_query)
    documents = rerank_documents_with_keywords(retrieved_docs, keywords)

    query_doc_pairs = [[synthesized_query, doc.page_content] for doc in documents]
    scores = reranker_model.predict(query_doc_pairs)
    scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
    
    top_5_docs = scored_docs[:15]

    final_results = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        }
        for score, doc in top_5_docs
    ]

    #mentos 중복 제거
    deduplicated_final_results = deduplicate_results(final_results)

    final_top_3 = deduplicated_final_results[:3]

    if member_seq in request.app.state.conversation_memory:
        del request.app.state.conversation_memory[member_seq]
        logger.info(f"사용자 {member_seq}의 챗봇 대화 기록이 초기화되었습니다.")

    #front에 뭐 보내줘야하는지 확인필요! title,mento,mento_profile,price,bname?
    return JSONResponse(
        content={"synthesized_query": synthesized_query, "documents": final_top_3},
        status_code=200,
    )

#각 쿼리를 순회돌아 invoke하여 후보군을 형성한 후 종합 쿼리(각 문장을 하나로 통합한 문장)로 re-rank
# @router.post("/recommend2")
# async def recommend(request: Request, queries: TextRequest, db: Session = Depends(get_db)):
#     model = request.app.state.model
#     llm_model = request.app.state.llm
#     reranker_model = request.app.state.reranker

#     if not model:
#         return JSONResponse(
#             {"result": "임베딩 모델이 로드되지 않았습니다."},
#             status_code=500,
#         )
#     if not llm_model:
#         return JSONResponse(
#             {"result": "LLM 모델이 로드되지 않았습니다."},
#             status_code=500,
#         )
#     if not reranker_model:
#         return JSONResponse(
#             {"result": "Reranker 모델이 로드되지 않았습니다."},
#             status_code=500,
#         )
    
#     llm_model = request.app.state.llm
#     mentos_df = select_mentos_data_to_df(db)
#     logger.info(f"DataFrame loaded. Shape: {mentos_df.shape}")
    
#     mentos_documents = make_document(mentos_df)
#     logger.info("Documents created successfully.")

#     mentos_vectorstore = vector_embedding(mentos_documents, model)
#     mentos_vector_retriever = make_vector_retriever(mentos_vectorstore)
#     Otk_retriever = make_bm25_retriever(mentos_documents, okt_tokenize)
#     mentos_ensemble_retriever = ensemble_retriever(mentos_vector_retriever, Otk_retriever, llm_model)
    
#     queries_list = queries.queries

#     all_retrieved_docs = {} # 중복 제거를 위해 dict 사용 (key: mentos_seq)
#     for query in queries_list:
#         retrieved_docs = mentos_ensemble_retriever.invoke(query)
#         for doc in retrieved_docs:
#             mentos_seq = doc.metadata.get("mentos_seq")
#             if mentos_seq and mentos_seq not in all_retrieved_docs:
#                 all_retrieved_docs[mentos_seq] = doc

#     unique_documents = list(all_retrieved_docs.values())

#     if not unique_documents:
#         return JSONResponse(
#             content={"synthesized_query": "N/A", "documents": []},
#             status_code=200,
#         )
    
#     synthesized_query = synthesize_query(queries_list, llm_model)

#     query_doc_pairs = [[synthesized_query, doc.page_content] for doc in unique_documents]
#     scores = reranker_model.predict(query_doc_pairs)
#     scored_docs = sorted(zip(scores, unique_documents), key=lambda x: x[0], reverse=True)
#     top_5_docs = scored_docs[:5]

#     final_results = [
#         {
#             "page_content": doc.page_content,
#             "metadata": doc.metadata,
#             "score": float(score)
#         }
#         for score, doc in top_5_docs
#     ]

#     return JSONResponse(
#         content={"synthesized_query": synthesized_query, "documents": final_results},
#         status_code=200,
#     )