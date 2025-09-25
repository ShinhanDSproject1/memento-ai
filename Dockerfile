FROM python:3.13.5-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-dev \
    openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64

RUN pip install --no-cache-dir numpy setuptools==80.9.0

RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .

EXPOSE 8001
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8001"]