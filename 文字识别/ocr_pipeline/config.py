"""OCR 流水线的集中配置。"""

from pathlib import Path


# vLLM 服务
API_BASE = "http://127.0.0.1:8000/v1"
MODEL = "Qwen3.6-27B"
API_KEY = "EMPTY"

DEFAULT_PDF_PATH = Path(
    "/home/huarui/pythonProject/data_generate/灵枢数据补充/知识图谱智能体/"
    "data/冠心病数据/冠心病搜集数据_V260721/1-指南/01_纯西医/"
    "瓣膜病/场景_围术期/成人瓣膜性心脏病围术期管理专家共识(2025年).pdf"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

# PDF 页面渲染
RENDER_ZOOM = 3.0
BLANK_CHECK_ZOOM = 0.5
BLANK_MARGIN_RATIO = 0.03
BLANK_INK_RATIO_THRESHOLD = 0.0015
BLANK_CONTRAST_THRESHOLD = 8.0
BLANK_BACKGROUND_DELTA = 25

# 原生电子排版检测。判断采取保守策略：不确定或混合版 PDF 走 OCR。
NATIVE_MIN_TEXT_CHARS = 50
NATIVE_MIN_TEXT_PAGE_RATIO = 0.80
FULL_PAGE_IMAGE_MIN_COVERAGE = 0.75
NATIVE_MAX_FULL_PAGE_IMAGE_RATIO = 0.10
MARGIN_HEIGHT_RATIO = 0.10

# 即使结构上是原生 PDF，编码异常或版面过于复杂时仍改走视觉 OCR。
NATIVE_MAX_FULLWIDTH_ASCII_RATIO = 0.10
NATIVE_MAX_PRIVATE_USE_RATIO = 0.0005
NATIVE_MAX_MEDIAN_TEXT_BLOCKS = 60

# “原生优先”阶段使用的高置信阈值。宁可把原生 PDF 留给 OCR，
# 也不直接提取编码异常、混合扫描或版面过于复杂的文件。
STRICT_NATIVE_MIN_TEXT_PAGE_RATIO = 0.98
STRICT_NATIVE_MAX_FULL_PAGE_IMAGE_RATIO = 0.0
STRICT_NATIVE_MAX_FULLWIDTH_ASCII_RATIO = 0.03
STRICT_NATIVE_MAX_PRIVATE_USE_RATIO = 0.0
STRICT_NATIVE_MAX_REPLACEMENT_CHAR_RATIO = 0.0
STRICT_NATIVE_MAX_MEDIAN_TEXT_BLOCKS = 40

START_PAGE = 1
END_PAGE: int | None = None
MAX_OUTPUT_TOKENS = 8192
MAX_WORKERS = 128
FILE_WORKERS = 1
MAX_TOTAL_CONCURRENCY = 128
MAX_NATIVE_PROCESSES = 64

# Qwen 采样参数。此处保留拆分前文件中的当前值。
TEMPERATURE = 0.1
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 1.5
REPETITION_PENALTY = 1.0

OCR_PROMPT = (
    "你是一个高精度文档 OCR 引擎。请先在内部判断页面的实际语言、文字方向、"
    "阅读顺序、分栏结构，以及是否包含标题、正文、列表、目录、表格、公式或图片说明，"
    "然后转录页面中全部可见文字。不要输出分析过程。\n"
    "要求：\n"
    "1. 根据页面实际版式确定阅读顺序；横排、竖排、单栏和多栏均可能出现。\n"
    "2. 使用 Markdown 尽量保留标题、自然段、列表和表格结构。"
    "同一自然段必须连续输出；"
    "只有真正切换自然段、标题、列表项或表格行时才换行。\n"
    "3. 不要输出页眉、页脚、独立页码、版心题名、扫描水印、文件路径等非正文内容；"
    "但属于正文结构的章节标题必须保留。\n"
    "4. 所有中文统一输出为简体中文。只进行繁体字到简体字的对应转换，"
    "不得改写原句、替换词语、调整语序、翻译、总结或补全；非中文内容保持原样。\n"
    "5. 忠实保留正文中的大小写、数字和标点。\n"
    "6. 目录中的装饰性点线可以省略，但必须保留条目名称及其对应页码。\n"
    "7. 无法辨认的单个字符用〔□〕表示，不要根据上下文猜测。\n"
    "8. 页面没有正文内容时输出空内容。\n"
    "只输出最终转录结果。"
)
