import re
from dataclasses import dataclass, field

from .thread_rng import rng
from typing import Dict, List, Optional, Tuple

PLACEHOLDER_RE = re.compile(r"\[([A-Z])\]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_generated_text(text: str) -> str:
    """Lightweight grammar cleanup after placeholder substitution."""
    text = WHITESPACE_RE.sub(" ", str(text or "")).strip()
    if not text:
        return ""
    # Preserve JSON/list answers such as 3D grounding targets.
    if text[0] in "[{":
        return text
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


@dataclass(frozen=True)
class PromptRenderRecord:
    """Provenance captured at template sample + fill time (single source of truth)."""

    template_id: str
    question_index: int
    answer_index: int
    question_line: str
    answer_line: str
    question_text: str
    answer_text: str
    question_bindings: Dict[str, str]
    answer_bindings: Dict[str, str]
    # M8 structured render (empty when using legacy flat PromptTemplate only)
    question_type: str = ""
    instruction_type: str = ""
    constraint_mode: str = ""
    introduction_index: int = -1
    question_instruction_index: int = -1

    def to_prompt(self) -> str:
        return f"{self.question_text} Answer: {self.answer_text}"


@dataclass
class PromptTemplate:
    """A group of question/answer templates."""
    questions: List[str]
    answers: List[str] = field(default_factory=list)
    true_answers: Optional[List[str]] = None
    false_answers: Optional[List[str]] = None

    def sample_pair(self, condition: bool = None) -> Tuple[int, int, str, str]:
        """Return (question_index, answer_index, question_line, answer_line)."""
        qi = rng().randrange(len(self.questions))
        q = self.questions[qi]
        if condition is not None:
            if self.true_answers and self.false_answers:
                pool = self.true_answers if condition else self.false_answers
                ai = rng().randrange(len(pool))
                a = pool[ai]
                answer_index = ai
            else:
                raise ValueError(
                    "sample(condition=...) requires both true_answers and "
                    "false_answers to be non-empty."
                )
        elif self.answers:
            ai = rng().randrange(len(self.answers))
            a = self.answers[ai]
            answer_index = ai
        else:
            answer_index = 0
            a = ""
        return qi, answer_index, q, a

    def sample(self, condition: bool = None) -> Tuple[str, str]:
        _, _, q, a = self.sample_pair(condition)
        return q, a

    @staticmethod
    def placeholders_in_line(line: str) -> List[str]:
        return PLACEHOLDER_RE.findall(line)

    @staticmethod
    def _fill(text: str, mapping: dict) -> str:
        for key, val in mapping.items():
            text = text.replace(f"[{key}]", str(val))
        return text

    @staticmethod
    def _bindings_applied_to_line(
        line: str,
        shared: Optional[dict],
        line_args: Optional[dict],
    ) -> Dict[str, str]:
        """Record placeholder values that apply to this line (shared then line-specific)."""
        keys = set(PLACEHOLDER_RE.findall(line))
        out: Dict[str, str] = {}
        for src in (shared, line_args):
            if not src:
                continue
            for key, val in src.items():
                if key in keys and val is not None:
                    out[key] = str(val)
        return out

    def render_provenance(
        self,
        template_id: str,
        condition: bool = None,
        *,
        shared: dict = None,
        q_args: dict = None,
        a_args: dict = None,
    ) -> PromptRenderRecord:
        """Sample template lines, fill placeholders, return structured provenance."""
        qi, ai, q_line, a_line = self.sample_pair(condition)
        q_text = q_line
        a_text = a_line
        if shared:
            q_text = self._fill(q_text, shared)
            a_text = self._fill(a_text, shared)
        if q_args:
            q_text = self._fill(q_text, q_args)
        if a_args:
            a_text = self._fill(a_text, a_args)
        return PromptRenderRecord(
            template_id=template_id,
            question_index=qi,
            answer_index=ai,
            question_line=q_line,
            answer_line=a_line,
            question_text=normalize_generated_text(q_text),
            answer_text=normalize_generated_text(a_text),
            question_bindings=self._bindings_applied_to_line(q_line, shared, q_args),
            answer_bindings=self._bindings_applied_to_line(a_line, shared, a_args),
        )

    def render(self, condition: bool = None, *,
               shared: dict = None, q_args: dict = None, a_args: dict = None) -> str:
        return self.render_provenance(
            "", condition, shared=shared, q_args=q_args, a_args=a_args,
        ).to_prompt()

    def render_qa(self, condition: bool = None, *,
                  shared: dict = None, q_args: dict = None, a_args: dict = None) -> Tuple[str, str]:
        rec = self.render_provenance(
            "", condition, shared=shared, q_args=q_args, a_args=a_args,
        )
        return rec.question_text, rec.answer_text


class TemplateRegistry:
    """Global template registry keyed by 'task.variant' names."""
    _store: Dict[str, PromptTemplate] = {}

    @classmethod
    def register(cls, name: str, tpl: PromptTemplate):
        cls._store[name] = tpl

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        if name not in cls._store:
            raise KeyError(f"Template '{name}' not registered. Available: {list(cls._store.keys())}")
        return cls._store[name]

    @classmethod
    def keys(cls) -> list:
        return list(cls._store.keys())
