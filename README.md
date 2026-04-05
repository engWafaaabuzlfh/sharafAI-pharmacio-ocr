# Pharmacio AI Microservice

The **Pharmacio AI Microservice** is an asynchronous Optical Character Recognition (OCR) engine that extracts structured text from pharmacy-related documents (such as distributor invoices) using computer vision and machine learning.

It runs as a decoupled FastAPI service and works securely in tandem with `pharmacio-backend`.

## 🚀 How it Works

When a user uploads a PDF in the backend, the backend triggers an OCR job on this microservice rather than blocking the main Django application.

1. **Job Acceptance**: The backend sends a `POST` request to `/v1/process` with a `job_id` and the `file_reference` (a storage path/key). The AI service immediately responds with `202 Accepted` to unblock the backend.
2. **File Fetching**: In a background task, the engine securely downloads the PDF file using the specified `STORAGE_BACKEND` (Local files or an S3 compatible bucket like AWS/MinIO).
3. **Pipeline Execution**: 
   - `pdf2image` converts the PDF into raw images.
   - Using OpenCV, the `integrated_pipeline` denoises the images and detects tabular grid cells based on morphological operations.
   - Text is read dynamically from those cells in Arabic and English using `EasyOCR`.
4. **Callback to Backend**: After formatting the extracted text into structured rows, the engine securely POSTs the result back to the Django backend's callback URL (`/api/v1/ocr/result/`).

## 🛠️ Tech Stack

- **Python 3.11**
- **FastAPI** + **Uvicorn**: Asynchronous HTTP API handling.
- **EasyOCR**: Model-based Arabic & English text extraction.
- **OpenCV** & **Pandas**: Image processing and table structuring.
- **Boto3**: For handling S3 bucket file storage fetches.

## 📦 Setup & Installation

The simplest way to run this microservice is using Docker. It requires system packages like `poppler-utils` and `libgl1` to process PDFs and OpenCV images properly.

```bash
docker compose up --build -d
```
*Note: This relies on the `.env` variables located in the `pharmacio-backend` directory.*

## ⚙️ Environment Variables

The OCR engine shares its configuration concepts with the backend to ensure secure dispatch and callback integrations.

| Variable | Description |
|---|---|
| `AI_ENGINE_API_KEY` | The secret key required to authorize the inbound `/v1/process` requests. |
| `INTERNAL_SERVICE_TOKEN` | The secret key used to compute the `Authorization` header when POSTing results back to the backend. |
| `BACKEND_BASE_URL` | Base URL of the backend (e.g., `http://backend:8000`). |
| `STORAGE_BACKEND` | Must match the backend. Accepts `s3` or `local`. |
| `LOCAL_MEDIA_ROOT` | Target Django media root path if using local storage. |

### AWS / S3 Configuration
If `STORAGE_BACKEND` is set to `s3`:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_S3_ENDPOINT_URL`
- `AWS_STORAGE_BUCKET_NAME`
- `AWS_S3_REGION_NAME`

## 📡 API Reference

### `POST /v1/process`

Accepts a new OCR task.

**Headers:**
```http
Authorization: <AI_ENGINE_API_KEY>
```

**Body:**
```json
{
  "job_id": "c1f71... (UUID)",
  "file_reference": "path/to/invoice.pdf"
}
```

**Response:**
```json
// Status: 202 Accepted
{
  "status": "accepted"
}
```

### `GET /health`
Returns the status of the service `{ "status": "ok" }`.
