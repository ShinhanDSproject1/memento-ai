import json
from app.utils.logger import setup_logger

logger = setup_logger()

in_memory_db = {
    "meaningful_sentences": []
}

def save_user_data(user_id: str, data: dict) -> str:
    global in_memory_db
    user_data = in_memory_db.get(user_id, {})
    user_data.update(data)
    in_memory_db[user_id] = user_data
    
    # Python 딕셔너리를 JSON 문자열로 변환합니다.
    json_data = json.dumps(data, ensure_ascii=False)
    
    logger.info(f"사용자 {user_id}의 프로필 데이터 저장 완료: {json_data}")
    
    # 변환된 JSON 문자열을 반환합니다.
    return f"사용자 정보 저장 완료: {json_data}"

def check_profile_complete(user_id: str) -> str:
    """추천에 필요한 모든 데이터가 수집되었는지 확인합니다."""
    global in_memory_db
    user_data = in_memory_db.get(user_id, {})
    
    # 'interest', 'level', 'goal' 세 가지 필수 키가 모두 있는지 확인합니다.
    required_keys = ['interest', 'level', 'goal']
    
    # 모든 필수 키가 존재하고 값이 None이 아닌지 한 번에 검사
    logger.info(f"사용자 {user_id}의 프로필 데이터 확인: {user_data}")
    is_complete = all(user_data.get(key) is not None for key in required_keys)
    
    # 기존 코드와 동일하게 문자열 'True' 또는 'False' 반환
    return str(is_complete)
    
def save_meaningful_data(user_id: str, text: str) -> str:
    """의미 있는 문장을 별도의 리스트에 저장합니다."""
    # user_id와 함께 의미 있는 문장을 저장합니다.
    global in_memory_db
    in_memory_db["meaningful_sentences"].append({"user_id": user_id, "text": text})
    
    print(f"\n[의미 있는 문장 저장] '{text}'\n")
    return "의미 있는 문장 저장 완료."
