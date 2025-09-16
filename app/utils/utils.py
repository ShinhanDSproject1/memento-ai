import re

def extract_user_utterances(history: str) -> list[str]:
    """
    전체 대화 기록 문자열에서 "사용자:"로 시작하는 문장들만 추출하여
    리스트로 반환합니다.

    Args:
        history (str): "사용자: ...\n챗봇: ...\n" 형태의 전체 대화 기록

    Returns:
        List[str]: 사용자의 발언만 담긴 문자열 리스트
    """
    # 정규표현식을 사용하여 "사용자: " 뒷부분의 내용만 추출
    user_utterances = re.findall(r"사용자: (.*)", history)
    return user_utterances

def deduplicate_results(documents: list[dict]) -> list[dict]:
    """
    'page_content'를 기준으로 문서 딕셔너리 리스트의 중복을 제거합니다.
    리스트에서 가장 먼저 나타나는 문서를 유지합니다.

    Args:
        documents: 문서 딕셔너리의 리스트. 
        각 딕셔너리는 'page_content' 키를 포함해야 합니다.

    Returns:
        중복이 제거된 문서 딕셔너리의 리스트.
    """
    seen_contents = set()
    unique_documents = []
    
    for doc in documents:
        # doc 딕셔너리에서 'page_content' 값을 가져옵니다.
        content = doc.get("page_content")
        
        # page_content가 처음 나타나는 경우에만 리스트에 추가합니다.
        if content not in seen_contents:
            unique_documents.append(doc)
            seen_contents.add(content)
            
    return unique_documents