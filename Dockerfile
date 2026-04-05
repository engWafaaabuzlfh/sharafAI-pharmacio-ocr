# OCR engine microservice — poppler required for pdf2image; GL libs for OpenCV/EasyOCR
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first so pip does not pull multi-GB CUDA wheels (EasyOCR works on CPU).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models (ar, en) avoiding download on every container run
RUN mkdir -p /root/.EasyOCR/model \
    && python -c "import easyocr; easyocr.Reader(['ar', 'en'], gpu=False)"

COPY integrated_pipeline.py .
COPY ocr_engine ./ocr_engine

ENV PYTHONPATH=/app
EXPOSE 8080

CMD ["uvicorn", "ocr_engine.main:app", "--host", "0.0.0.0", "--port", "8080"]
