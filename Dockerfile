# 앱 버전을 위한 빌드 인수 (Build argument for app version)
ARG APP_VERSION="1.0.0"

# =================================================================
# 1. 빌드 스테이지 (Build Stage)
# - 파이썬 패키지를 설치하고 컴파일하는 역할만 수행합니다.
# =================================================================
FROM python:3.13.5-slim AS builder

WORKDIR /app

# 빌드에 필요한 시스템 의존성 설치
# openjdk-17-jdk를 openjdk-17-jre-headless로 변경 (더 가벼움)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-17-jre-headless \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 가상 환경 생성 (더 깔끔한 관리를 위함)
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 운영용 requirements-prod.txt를 복사하여 설치
COPY requirements-prod.txt .

# requirements-prod.txt에 명시된 패키지 설치
RUN pip install --no-cache-dir numpy setuptools==80.9.0
RUN pip install --no-cache-dir --no-build-isolation -r requirements-prod.txt


# =================================================================
# 2. 최종 스테이지 (Final Stage)
# - 실제 애플리케이션을 실행하는 역할만 수행합니다.
# =================================================================
FROM python:3.13.5-slim

# 최종 스테이지에서 ARG를 다시 선언하여 변수 사용
ARG APP_VERSION

# 이미지 메타데이터로 버전 정보 추가
LABEL version=$APP_VERSION

# 컨테이너 내부에서 사용할 환경 변수로 버전 설정
ENV APP_VERSION=$APP_VERSION

WORKDIR /app

# 런타임에만 필요한 시스템 의존성 설치
# openjdk-17-jdk를 openjdk-17-jre-headless로 변경
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    openjdk-17-jre-headless \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Java 환경 변수 설정 (jre 경로로 변경)
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-arm64/jre"

# 빌드 스테이지에서 설치한 파이썬 패키지(가상 환경)를 그대로 복사
COPY --from=builder /opt/venv /opt/venv

# 애플리케이션 코드를 복사
COPY . .

# 가상 환경의 실행 파일을 사용하도록 PATH 설정
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8001
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8001"]

