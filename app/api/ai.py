from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from app.schemas.text_schema import TextRequest
from app.schemas.message import Message
from app.db.database import get_db
import pandas as pd
from fastapi.responses import JSONResponse
from app.utils.logger import setup_logger
from app.core.recommend.chat_bot import LLMAgentService, AgentExecutionError
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
    synthesize_query,
    build_ensemble_retrieve_re,
)

logger = setup_logger()
router = APIRouter()
llm_service = LLMAgentService()
conversation_memory = {}
    
@router.get("/chatbot")
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
async def chat_with_bot(request: Request, message:Message):
    llm_model = request.app.state.llm
    system_message = request.app.state.system_message

    if not llm_model:
        return JSONResponse(
                {"error": "LLM 모델이 로드되지 않았습니다."},
                status_code=500,
        )
    
    user_id = message.user_id
    user_input = message.content

    history = conversation_memory.get(user_id, "")

    try:
        response_from_agent, is_ready = await llm_service.run_agent_flow(user_id, user_input, llm_model, history, system_message)

        new_entry = f"사용자: {user_input}\n챗봇: {response_from_agent}\n"
        conversation_memory[user_id] = history + new_entry
        
        return JSONResponse(
            {
                "response": response_from_agent,
                "recommendation_ready": is_ready,
            },
            status_code=200,
        )

    except AgentExecutionError as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )

#종합 쿼리(각 문장을 하나로 통합한 문장)로 invoke후 re-rank
@router.post("/recommend")
async def recommend2(request: Request, queries: TextRequest, db: Session = Depends(get_db)):
    model = request.app.state.model
    llm_model = request.app.state.llm
    reranker_model = request.app.state.reranker
    if not model:
        return JSONResponse(
            {"result": "임베딩 모델이 로드되지 않았습니다."},
            status_code=500,
        )
    if not llm_model:
        return JSONResponse(
            {"result": "LLM 모델이 로드되지 않았습니다."},
            status_code=500,
        )
    if not reranker_model:
        return JSONResponse(
            {"result": "Reranker 모델이 로드되지 않았습니다."},
            status_code=500,
        )

    mentos_df = select_mentos_data_to_df(db)
    logger.info(f"DataFrame loaded. Shape: {mentos_df.shape}")
    
    mentos_documents = make_document(mentos_df)
    logger.info("Documents created successfully.")

    mentos_vectorstore = vector_embedding(mentos_documents, model)
    mentos_vector_retriever = make_vector_retriever(mentos_vectorstore)
    Otk_retriever = make_bm25_retriever(mentos_documents, okt_tokenize)
    mentos_ensemble_retriever = build_ensemble_retrieve_re(mentos_vector_retriever, Otk_retriever)
    
    queries_list = queries.queries
    synthesized_query = synthesize_query(queries_list, llm_model)
    retrieved_docs = mentos_ensemble_retriever.invoke(synthesized_query)
    query_doc_pairs = [[synthesized_query, doc.page_content] for doc in retrieved_docs]
    scores = reranker_model.predict(query_doc_pairs)
    scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
    top_5_docs = scored_docs[:5]

    final_results = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        }
        for score, doc in top_5_docs
    ]

    return JSONResponse(
        content={"synthesized_query": synthesized_query, "documents": final_results},
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