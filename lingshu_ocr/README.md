# PDF 文字识别

Qwen 入口对所有 PDF 页面统一执行视觉 OCR：

- 不判断或读取 PDF 原生文本层；
- 原生、扫描、可搜索扫描及混合 PDF 均调用 Qwen3.6-27B OCR；
- 默认启动 4 个 PDF 工作进程，每个进程内部 128 页线程并发，
  总并发不超过 512；
- 运行时只显示一个汇总所有 PDF 页面的 `tqdm` 进度条；
- Qwen 子进程每完成一页都会通知主进程实时更新进度；
- 所有页面按页码顺序写入同一个 TXT 或 Markdown 文件，默认使用 TXT。

也可以先用更严格的高置信规则直接提取原生 PDF，再让 Qwen 处理剩余文件：

```bash
# 第一阶段：只直接提取高置信原生 PDF
python ocr_qwen.py /path/to/data --recursive --native-only \
    --file-workers 32 --format txt

# 第二阶段：已有结果自动跳过，其余文件全部使用 Qwen OCR
python ocr_qwen.py /path/to/data --recursive \
    --file-workers 4 --workers 128 --format txt
```

## 运行

```bash
conda activate data_generate
python ocr_qwen.py file.pdf
python ocr_qwen.py /home/huarui/pythonProject/data_generate/灵枢数据补充/知识图谱智能体/data --recursive --file-workers 4 --workers 128 --format txt
```

DeepSeek-OCR-2 八端口版本：

```bash
python ocr_deepseek.py /home/huarui/pythonProject/data_generate/灵枢数据补充/知识图谱智能体/data \
    --recursive \
    --file-workers 8 \
    --workers 128 \
    --per-server-concurrency 128 \
    --format txt
```

默认服务地址为 `8080-8087`，客户端总并发上限为
`8 个端口 × 128 = 1024`。DeepSeek 版本强制所有 PDF 页面走 OCR，
不读取 PDF 原生文本层；结果保存到 `output/ocr_deepseek`。

## 模块

- `ocr_qwen.py`：精简命令行入口；
- `ocr_pipeline/config.py`：服务地址、模型、采样参数、并发数和提示词；
- `ocr_pipeline/pdf_detection.py`：原生电子排版检测；
- `ocr_pipeline/native_extractor.py`：直接读取、页眉页脚清理、段落拼接；
- `ocr_pipeline/qwen_ocr.py`：页面渲染、空白页检测、Qwen 请求；
- `ocr_pipeline/text_cleaning.py`：视觉换行与退化重复清理；
- `ocr_pipeline/processor.py`：自动分流、页级并发和有序写入；
- `ocr_pipeline/cli.py`：批量文件和命令行参数。

## 简繁转换

OCR 提示词要求输出简体。原生 PDF 直接提取后，如需强制转换简体，请安装：

```bash
pip install opencc-python-reimplemented
```


python ocr_qwen.py /home/huarui/pythonProject/data_generate/灵枢数据补充/知识图谱智能体/data \
    --recursive \
    --file-workers 32 \
    --workers 16 \
    --format txt



python 文字识别/ocr_deepseek.py data \
    --recursive \
    --file-workers 8 \
    --workers 128 \
    --per-server-concurrency 128 \
    --format txt
