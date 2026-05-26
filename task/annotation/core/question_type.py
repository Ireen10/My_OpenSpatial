from enum import Enum


class QuestionType(str, Enum):
    """Question type enum. Inherits str so serialization to parquet is transparent."""

    OPEN_ENDED = "open_ended"  # 开放式问答
    MCQ = "MCQ"  # 选择题
    JUDGMENT = "judgment"  # 判断题（是/否等）
