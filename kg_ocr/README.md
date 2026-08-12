# PDF OCR

## Run

```bash
conda activate data_generate
python ocr_qwen.py file.pdf
python ocr_qwen.py data --recursive --file-workers 4 --workers 128 --format txt
```

DeepSeek-OCR-2:

```bash
python ocr_deepseek.py data \
    --recursive \
    --file-workers 8 \
    --workers 128 \
    --per-server-concurrency 128 \
    --format txt
```

## Modules

- `ocr_qwen.py`: thin CLI entry point
- `ocr_pipeline/config.py`: service URL, model, sampling, concurrency, and prompts
- `ocr_pipeline/pdf_detection.py`: native digital-layout detection
- `ocr_pipeline/native_extractor.py`: direct text extract, header/footer cleanup, paragraph joining
- `ocr_pipeline/qwen_ocr.py`: page render, blank-page check, Qwen requests
- `ocr_pipeline/text_cleaning.py`: visual line-break and degenerate-repetition cleanup
- `ocr_pipeline/processor.py`: routing, page-level concurrency, ordered writes
- `ocr_pipeline/cli.py`: batch files and CLI arguments

## Simplified Chinese conversion

The OCR prompt asks for Simplified Chinese output. After native PDF extraction, install the following package if you need forced conversion to Simplified Chinese:

```bash
pip install opencc-python-reimplemented
```
