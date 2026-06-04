# OCR engine microservice — poppler required for pdf2image; GL libs for OpenCV/EasyOCR
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models (ar, en) avoiding download on every container run
RUN mkdir -p /root/.EasyOCR/model \
    && python -c "import easyocr; easyocr.Reader(['ar', 'en'], gpu=False)"

COPY run.py .
COPY API_Engine ./API_Engine
COPY common ./common
COPY services ./services
COPY integrated_pipeline.py .

ENV PYTHONPATH=/app
EXPOSE 8080

CMD ["python", "run.py"]
