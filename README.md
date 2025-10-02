# MEMENTO-AI
신한DS 금융SW아카데미 5기 2회차 Team 메멘토의 FASTAPI기반 AI 및 OCR 레포지토리입니다. :smiley: <br>
:warning: 본 프로젝트는 개인의 AWS, OPENAI API KEY와 Huggingface HF_TOKEN 및 Mysql 관련 환경변수를 .env파일에 등록해주셔야 동작합니다!
:warning: 현재 uvloop 라이브러리로 인해 Mac과 리눅스 환경에서만 동작 가능합니다...(WSL 환경 세팅 방법 추가 예정)
# 📦 Install
```
git clone https://github.com/ShinhanDSproject1/memento-ai.git
```
- Python link: [Python-3.13.5][PythonLink]
- venv 참고blog: [호무비 파이썬 가상환경(venv) 종류 및 사용법 정리][VenvLink]
- TesseractOCR: [TesseractOCR-Github][VenvLink]
- TesseractOCR 참고blog: [콩다코딩 OCRTesseract OCR 설치 및 사용방법][TesseractHelpLink]
- JDK-17 link: [JDK-17][JDK-17Link] -> 한국어 형태소 분석기(KKMA, Okt) 사용하기에 필요
```
pip install -r requirements.txt
```
# 🚀 FrameWork
- FastAPI link: [FastAPI][FastAPILink]
- Pytorch link: [Pytorch][PytorchLink]

[PythonLink]: https://www.python.org/downloads/release/python-3135/
[VenvLink]: https://homubee.tistory.com/38
[TesseractOCRLink]: https://github.com/UB-Mannheim/tesseract/wiki
[TesseractHelpLink]: https://kongda.tistory.com/93
[FastAPILink]: https://fastapi.tiangolo.com/ko/
[JDK-17Link]: https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html
[PytorchLink]: https://pytorch.org/get-started/locally/
# :computer: CMD commend
#### For dev
```
uvicorn app.main:app --reload
```
#### For local-prod
```
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

👉 **배포용 DockerFile도 활용가능** <br>
👉[세팅 참고 노션페이지][NotionLink]

[NotionLink]: https://unleashed-loan-37c.notion.site/?source=copy_link



