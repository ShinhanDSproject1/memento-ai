import re

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