import re
import pandas as pd
from kiwipiepy import Kiwi
from collections import Counter

kiwi = Kiwi()

def deduplicate_results(documents: list[dict]) -> list[dict]:
    """
    'page_content'를 정규화하여 중복을 제거합니다.
    """
    seen_contents = set()
    unique_documents = []
    
    for doc in documents:
        content = doc.get("page_content")
        
        # 1. 모든 공백을 하나의 공백으로 통일하고
        # 2. 양쪽 끝 공백을 제거
        normalized_content = ' '.join(content.split()).strip()
        
        if normalized_content not in seen_contents:
            unique_documents.append(doc)
            seen_contents.add(normalized_content)
            
    return unique_documents

def find_boilerplate_phrases(df, content_column='content', threshold_ratio=0.3):
    """
    kiwipiepy를 사용하여 반복되는 템플릿 문구를 찾습니다.
    """
    all_sentences = []
    for text in df[content_column].dropna():
        # kiwi.split_into_sents는 Sentence 객체의 리스트를 반환합니다.
        # .text를 이용해 실제 텍스트만 추출합니다.
        sentences = [s.text for s in kiwi.split_into_sents(str(text))]
        cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        all_sentences.extend(cleaned_sentences)
        
    # --- 이하 로직은 이전과 완전히 동일합니다 ---
    sentence_counts = Counter(all_sentences)
    total_documents = len(df)
    threshold_count = int(total_documents * threshold_ratio)
    
    boilerplate_candidates = []
    for sentence, count in sentence_counts.most_common():
        if count >= threshold_count:
            boilerplate_candidates.append(sentence)
        else:
            break
            
    return boilerplate_candidates

def remove_phrases_from_text(text: str, phrases_to_remove: list[str]) -> str:
    """주어진 텍스트에서 제거할 문구 리스트에 포함된 모든 문장을 제거합니다."""
    if not isinstance(text, str): # 텍스트가 아닌 경우(e.g., NaN) 빈 문자열 반환
        return ""
        
    cleaned_text = text
    for phrase in phrases_to_remove:
        cleaned_text = cleaned_text.replace(phrase, "")
    
    # 여러 줄바꿈과 공백을 정리
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()
    return cleaned_text