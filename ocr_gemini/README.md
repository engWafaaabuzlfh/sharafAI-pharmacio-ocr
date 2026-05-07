# Gemini OCR Option

This option sends the PDF directly to Gemini and asks it to return structured JSON.

## Install

```powershell
pip install google-genai
```

## Environment

Add one of these to `.env`:

```env
GOOGLE_API_KEY=your-google-api-key
```

or:

```env
GEMINI_API_KEY=your-google-api-key
```

## Run

```powershell
python -m ocr_gemini.extractor media/invoices/test.pdf -o gemini_result.json
```

The result is printed and optionally written to the output JSON file.
