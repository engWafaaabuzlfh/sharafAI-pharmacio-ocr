# Hugging Face Qwen2-VL OCR Option

This option uses `Qwen/Qwen2-VL-7B-Instruct` locally as a vision-language document extraction model.

It renders PDF pages with PyMuPDF, sends each page image to Qwen2-VL with a JSON-focused prompt, and writes one structured JSON result.

## Why Qwen2-VL?

- Strong document/image understanding.
- Better suited for Arabic + English mixed documents than many invoice-only models.
- Can follow a prompt and produce JSON directly.
- Does not require Poppler because PDF rendering uses PyMuPDF.

## Requirements

This is a large model. It is best with a GPU. CPU execution can be extremely slow.

Install:

```powershell
pip install transformers accelerate qwen-vl-utils pymupdf pillow torch torchvision
```

If your installed `transformers` version does not support Qwen2-VL, install a newer version:

```powershell
pip install -U transformers accelerate qwen-vl-utils
```

## Run

```powershell
python -m ocr_huggingFace.extractor media/invoices/test.pdf -o hf_qwen_result.json
```

## Optional: Qwen2.5-VL

If your environment supports it, you can try the newer model:

```powershell
python -m ocr_huggingFace.extractor media/invoices/test.pdf -o hf_qwen25_result.json --model Qwen/Qwen2.5-VL-7B-Instruct
```

## Output Shape

```json
{
  "engine": "huggingface-qwen-vl",
  "model": "Qwen/Qwen2-VL-7B-Instruct",
  "pages": [
    {
      "page": 1,
      "document_type": "invoice",
      "language": "mixed",
      "metadata": {},
      "tables": [],
      "line_items": [],
      "raw_text": "",
      "warnings": []
    }
  ]
}
```
