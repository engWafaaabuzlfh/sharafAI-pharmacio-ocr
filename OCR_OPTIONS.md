# OCR Options

The project now has three OCR/document extraction options.

## 1. EasyOCR Pipeline

Best when you want the current OpenCV cell-detection pipeline.

```powershell
python -c "from ocr_pipeline.pipeline import process_pdf_to_json; print(process_pdf_to_json('media/invoices/test.pdf'))"
```

Requires Poppler for PDF conversion when running locally.

## 2. Gemini PDF Extraction

Best for fast experimentation and direct PDF-to-JSON extraction.

```powershell
pip install google-genai
python -m ocr_gemini.extractor media/invoices/test.pdf -o gemini_result.json
```

Requires:

```env
GOOGLE_API_KEY=your-key
```

or:

```env
GEMINI_API_KEY=your-key
```

## 3. Hugging Face Qwen2-VL Local Extraction

Best for a strong local vision-language model option without sending files to an external API.

```powershell
pip install transformers accelerate qwen-vl-utils pymupdf pillow torch torchvision
python -m ocr_huggingFace.extractor media/invoices/test.pdf -o hf_qwen_result.json
```

Default model:

```text
Qwen/Qwen2-VL-7B-Instruct
```

This uses PyMuPDF to render the PDF, so it does not require Poppler.

You can also try the newer model:

```powershell
python -m ocr_huggingFace.extractor media/invoices/test.pdf -o hf_qwen25_result.json --model Qwen/Qwen2.5-VL-7B-Instruct
```
