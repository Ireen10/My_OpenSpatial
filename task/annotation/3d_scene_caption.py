"""
3D scene caption annotation task: generate spatial scene descriptions via LLM.

Dataset human prompts use role+task only; API prompts use all modules.
Required config keys: api_key, base_url, model.
"""

from task.annotation.core.thread_rng import rng
import time
import base64
from PIL import Image
from openai import OpenAI

from .core.base_annotation_task import BaseAnnotationTask
from .core.message_placeholders import sync_messages_with_qa_images
from .core.prompt_template import PromptRenderRecord
from .core.question_type import QuestionType
from ..prompt_templates.caption_prompt_templates import (
    CAPTION_API_MODULES,
    CAPTION_DATASET_KEYS,
    CAPTION_DATASET_MODULES,
    CAPTION_DEFAULT_DROPOUT,
)
from utils.image_utils import convert_pil_to_bytes

REQUIRED_KEYS = ("api_key", "base_url", "model")


class CaptionGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Singleview 3D Scene Caption"
    DATASET_TEMPLATE_ID = "caption.open_ended"

    def __init__(self, args):
        super().__init__(args)
        missing = [k for k in REQUIRED_KEYS if k not in args]
        if missing:
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")
        self.client = OpenAI(api_key=args["api_key"], base_url=args["base_url"])
        self.model = args["model"]
        self.max_retries = args.get("max_retries", 5)
        self.retry_delay = args.get("retry_delay", 5)

    def check_example(self, example) -> bool:
        return "image" in example

    def _call_api(self, prompt, image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    stream=False,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpg;base64,{b64}",
                            }},
                        ],
                    }],
                )
                return resp.choices[0].message.content
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return None

    @staticmethod
    def _sample_modules(modules, dropout, keys_filter=None):
        parts = []
        indices = {}
        for name, pool in modules.items():
            if keys_filter is not None and name not in keys_filter:
                continue
            if rng().random() < dropout.get(name, 0.0):
                continue
            idx = rng().randrange(len(pool))
            parts.append(pool[idx])
            indices[name] = idx
        return parts, indices

    @classmethod
    def sample_dataset_prompt(cls, dropout=None):
        if dropout is None:
            dropout = CAPTION_DEFAULT_DROPOUT
        parts, indices = cls._sample_modules(
            CAPTION_DATASET_MODULES, dropout, keys_filter=CAPTION_DATASET_KEYS,
        )
        if not parts:
            role_i = rng().randrange(len(CAPTION_DATASET_MODULES["role"]))
            task_i = rng().randrange(len(CAPTION_DATASET_MODULES["task"]))
            parts = [
                CAPTION_DATASET_MODULES["role"][role_i],
                CAPTION_DATASET_MODULES["task"][task_i],
            ]
            indices = {"role": role_i, "task": task_i}
        return " ".join(parts), parts, indices

    @classmethod
    def sample_api_prompt(cls, dropout=None):
        if dropout is None:
            dropout = CAPTION_DEFAULT_DROPOUT
        parts, _ = cls._sample_modules(CAPTION_API_MODULES, dropout)
        if not parts:
            parts = [
                rng().choice(CAPTION_API_MODULES["role"]),
                rng().choice(CAPTION_API_MODULES["task"]),
            ]
        return " ".join(parts)

    def _dataset_render_record(self, question_text: str, parts: list) -> PromptRenderRecord:
        intro = parts[0].strip() if len(parts) > 1 else ""
        stem = parts[-1].strip() if parts else question_text.strip()
        return PromptRenderRecord(
            template_id=self.DATASET_TEMPLATE_ID,
            question_index=-1,
            answer_index=-1,
            question_line=stem,
            answer_line="",
            question_text=question_text.strip(),
            answer_text="",
            question_bindings={},
            answer_bindings={},
            question_type=QuestionType.OPEN_ENDED.value,
            instruction_type="generative",
            introduction_text=intro,
            question_stem_text=stem,
        )

    def apply_transform(self, example):
        if not self.check_example(example):
            return None, False

        image_path = example["image"]
        question_prompt, prompt_parts, _indices = self.sample_dataset_prompt()
        api_prompt = self.sample_api_prompt()
        caption = self._call_api(api_prompt, image_path)
        if caption is None:
            return None, False

        self._thread_local.last_prompt_render = self._dataset_render_record(
            question_prompt, prompt_parts,
        )

        qa_images = [{"bytes": convert_pil_to_bytes(Image.open(image_path))}]
        messages = [[
            {"from": "human", "value": question_prompt},
            {"from": "gpt", "value": caption},
        ]]
        example["messages"] = sync_messages_with_qa_images(messages, qa_images)
        example["QA_images"] = qa_images
        example["question_tags"] = [[self.QUESTION_TAG]]
        example["question_types"] = [QuestionType.OPEN_ENDED]
        return example, True
