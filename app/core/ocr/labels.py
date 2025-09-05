import os, re

ALLOWED_CERT_CODES = {"inforproc", "adsp", "sqld"}
THRESHOLD = 0.95

STAMP_TEMPLATES = {
    "inforproc": os.path.join("assets", "gisa.png"),
    "adsp": os.path.join("assets", "adsp.png"),
    "sqld": os.path.join("assets", "adsp.png"),
}

CERT_NAME_MAP = {
    "inforproc": "정보처리기사",
    "adsp": "데이터 분석 준전문가",
    "sqld": "SQL 개발자",
}

LABEL_PATTERNS_INFORPROC = {
    "자격번호": r"자\s*격\s*번\s*호",
    "자격종목": r"자\s*격\s*종\s*목",
    "성명": r"성\s*명|성\s*\s*\s*\s*명",
    "생년월일": r"생\s*년\s*월\s*일",
    "합격 연월일": r"합\s*격\s*연\s*월\s*일",
    "발급 연월일": r"발\s*급\s*연\s*월\s*일",
    "관리번호": r"관\s*리\s*번\s*호",
}

LABEL_PATTERNS_ADSP = {
    "종목 및 등급": r"종\s*목\s*및\s*등\s*급",
    "자격번호": r"자\s*격\s*번\s*호",
    "성명": r"성\s*명",
    "생년월일": r"생\s*년\s*월\s*일",
    "합격일자": r"합\s*격\s*일\s*자",
    "유효기간": r"유\s*효\s*기\s*간",
    "발급일자": r"발\s*급\s*일\s*자",
}

LABEL_PATTERNS_SQLD = {
    "종목 및 등급": r"종\s*목\s*및\s*등\s*급",
    "자격번호": r"자\s*격\s*번\s*호",
    "성명": r"성\s*명",
    "생년월일": r"생\s*년\s*월\s*일",
    "합격일자": r"합\s*격\s*일\s*자",
    "유효기간": r"유\s*효\s*기\s*간",
    "발급일자": r"발\s*급\s*일\s*자",
}

REQUIRED_FIELDS = {
    "inforproc": [
        "자격번호",
        "자격종목",
        "성명",
        "생년월일",
        "합격 연월일",
        "발급 연월일",
    ],
    "adsp": [
        "종목 및 등급",
        "자격번호",
        "성명",
        "생년월일",
        "합격일자",
        "유효기간",
        "발급일자",
    ],
    "sqld": [
        "종목 및 등급",
        "자격번호",
        "성명",
        "생년월일",
        "합격일자",
        "유효기간",
        "발급일자",
    ],
}


def normalize_text(s: str) -> str:
    import re

    s = s.replace(":", "").replace("：", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compile_label_patterns(label_patterns: dict):
    # 라벨 말미 공백/개행 등 방지
    return {k: re.compile(v + r"\s*$") for k, v in label_patterns.items()}
