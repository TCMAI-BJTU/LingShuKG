"""DeepSeek-OCR-2 八端口全量 PDF 文字识别入口。

默认使用 8080-8087，共 8 个服务；每个端口最多 128 个并发请求。
所有 PDF 页面均强制使用 OCR，不读取 PDF 原生文本层。
"""

from ocr_pipeline.deepseek_cli import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
