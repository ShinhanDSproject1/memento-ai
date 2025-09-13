from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from app.schemas.text_schema import TextRequest
from app.db.database import get_db

from app.core.recommend.faiss import(
    make_document,
    vector_embedding,
    reciprocal_rank_fusion,
    make_vector_retriever
)

router = APIRouter()

@router.post("/recommend")
async def recommend(request: Request, data: TextRequest, db: Session = Depends(get_db)):
    model = request.app.state.model
    result = db.execute("SELECT * FROM mentos where state = activate").fetchall()
    print(result)
    llm_model = request.app.state.llm

    if not model:
        return {"error": "모델이 로드되지 않았습니다."}
    
@router.get("/chat")
async def chatbot_by_gpt(request: Request):
    llm_model = request.app.state.llm
    system_message = request.app.state.system_message

    if not llm_model:
        return {"error": "모델이 로드되지 않았습니다."}
    
    result = llm_model.predict(
                message="자기소개 부탁해!",
                system_message=system_message + "대답은 '안녕하세요! 당신의 똑똑한 금융 친구 토리예요. 저와의 대화를 통해 복잡한 금융 이야기를 재미있게 풀어가고, 당신에게 꼭 맞는 서비스를 찾아 드릴게요. 지금 혹시 가장 고민하고 있거나 궁금한 점이 있으신가요? 저와의 대화를 통해 복잡한 금융 이야기를 재미있게 풀어가고, 당신에게 꼭 맞는 서비스를 찾아 드릴게요. 지금 혹시 가장 고민하고 있거나 궁금한 점이 있으신가요?'로 하기",
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                api_name="/chat"
        )