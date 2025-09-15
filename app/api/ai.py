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
    reciprocal_rank_fusion,
    make_vector_retriever,
    select_mentos_data_to_df
)

logger = setup_logger()
router = APIRouter()
llm_service = LLMAgentService()
conversation_memory = {}

@router.post("/recommend")
async def recommend(request: Request, queries: TextRequest, db: Session = Depends(get_db)):
    model = request.app.state.model
    if not model:
        return JSONResponse(
            {"result": "임베딩 모델이 로드되지 않았습니다."},
            status_code=500,
        )
    llm_model = request.app.state.llm
    mentos_df = select_mentos_data_to_df(db)
    logger.info(f"DataFrame loaded. Shape: {mentos_df.shape}")
    
    
    mentos_data_list = mentos_df.to_dict('records')
    logger.info(f"DataFrame converted to list of dicts. Number of records: {len(mentos_data_list)}")
    
    mentos_document = make_document(mentos_df)
    logger.info("Documents created successfully.")

    return JSONResponse(
        {
            "data": mentos_data_list,
        },
        status_code=200,
    )
    
@router.get("/chatbot")
async def chatbot_by_gpt(request: Request):
    llm_model = request.app.state.llm
    system_message = request.app.state.system_message

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