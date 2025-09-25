import re
import pandas as pd
from konlpy.tag import Kkma
from collections import Counter

kkma = Kkma()

def find_boilerplate_phrases(df, content_column='content', threshold_ratio=0.3):
    all_sentences = []
    # .dropna()를 통해 비어있는 행을 안전하게 건너뜁니다.
    for text in df[content_column].dropna():
        sentences = kkma.sentences(str(text))
        
        # 너무 짧은 문장은 제외하고 리스트에 추가합니다.
        cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        all_sentences.extend(cleaned_sentences)
        
    sentence_counts = Counter(all_sentences)
    total_documents = len(df)
    
    # 설정한 비율(threshold_ratio) 이상 나타나는 문장만 후보로 선택합니다.
    threshold_count = int(total_documents * threshold_ratio)
    
    boilerplate_candidates = []
    # 빈도가 높은 순으로 정렬하여 확인합니다.
    for sentence, count in sentence_counts.most_common():
        if count >= threshold_count:
            boilerplate_candidates.append(sentence)
        else:
            # 빈도 순으로 정렬했으므로, 임계값보다 낮은 문장이 나오면 중단합니다.
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