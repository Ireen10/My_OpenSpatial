"""
Structured prompt templates (M8 L1).

Per QuestionType:
  - Question: introduction (optional) + stem (required) + question_instruction (optional)
  - Answer: instruction_type -> instruction_snippets[] + answer_templates[]

Instruction types and pools are registered by downstream tasks only.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .prompt_template import PLACEHOLDER_RE, PromptRenderRecord, PromptTemplate
from .question_type import QuestionType


def _pick_optional(pool: Sequence[str]) -> tuple[int, str]:
    if not pool:
        return -1, ""
    idx = random.randrange(len(pool))
    return idx, pool[idx].strip()


def _pick_required(pool: Sequence[str], *, label: str) -> tuple[int, str]:
    if not pool:
        raise ValueError(f"{label} pool must be non-empty")
    idx = random.randrange(len(pool))
    return idx, pool[idx].strip()


def _fill(text: str, mapping: Optional[dict]) -> str:
    if not mapping:
        return text
    out = text
    for key, val in mapping.items():
        if val is not None:
            out = out.replace(f"[{key}]", str(val))
    return out


def _bindings_for_line(line: str, shared: Optional[dict], line_args: Optional[dict]) -> Dict[str, str]:
    keys = set(PLACEHOLDER_RE.findall(line))
    out: Dict[str, str] = {}
    for src in (shared, line_args):
        if not src:
            continue
        for key, val in src.items():
            if key in keys and val is not None:
                out[key] = str(val)
    return out


@dataclass
class AnswerInstructionProfile:
    """One answer instruction_type: how the model should format the reply."""

    instruction_type: str
    instruction_snippets: List[str] = field(default_factory=list)
    answer_templates: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.answer_templates:
            raise ValueError(
                f"instruction_type {self.instruction_type!r}: answer_templates must be non-empty"
            )


@dataclass
class StructuredPromptTemplate:
    """
    Template bundle for a single template_id and QuestionType.

    Task registers pools; may restrict which answer instruction_types are enabled
    via enabled_answer_instruction_types (defaults to all registered profiles).
    """

    template_id: str
    question_type: QuestionType
    stem: List[str]
    introduction: List[str] = field(default_factory=list)
    question_instruction: List[str] = field(default_factory=list)
    answer_profiles: Dict[str, AnswerInstructionProfile] = field(default_factory=dict)
    enabled_answer_instruction_types: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if not self.stem:
            raise ValueError(f"{self.template_id}: stem pool must be non-empty")
        for itype, prof in self.answer_profiles.items():
            if prof.instruction_type != itype:
                raise ValueError(
                    f"{self.template_id}: profile key {itype!r} != "
                    f"profile.instruction_type {prof.instruction_type!r}"
                )
        enabled = self.enabled_answer_instruction_types
        if enabled is not None:
            missing = [t for t in enabled if t not in self.answer_profiles]
            if missing:
                raise ValueError(
                    f"{self.template_id}: enabled_answer_instruction_types "
                    f"not registered: {missing}"
                )

    def enabled_types(self) -> List[str]:
        if self.enabled_answer_instruction_types is not None:
            return list(self.enabled_answer_instruction_types)
        return list(self.answer_profiles.keys())

    def render_provenance(
        self,
        *,
        condition: Optional[bool] = None,
        instruction_type: Optional[str] = None,
        stem_index: Optional[int] = None,
        shared: Optional[dict] = None,
        q_args: Optional[dict] = None,
        a_args: Optional[dict] = None,
    ) -> PromptRenderRecord:
        intro_i, intro_line = _pick_optional(self.introduction)
        if stem_index is not None:
            if stem_index < 0 or stem_index >= len(self.stem):
                raise IndexError(
                    f"{self.template_id}: stem_index {stem_index} out of range "
                    f"(0..{len(self.stem) - 1})"
                )
            stem_i, stem_line = stem_index, self.stem[stem_index]
        else:
            stem_i, stem_line = _pick_required(self.stem, label="stem")
        qinstr_i, qinstr_line = _pick_optional(self.question_instruction)

        types = self.enabled_types()
        if not types:
            raise ValueError(f"{self.template_id}: no answer instruction_type enabled")
        itype = instruction_type or random.choice(types)
        if itype not in self.answer_profiles:
            raise KeyError(
                f"{self.template_id}: instruction_type {itype!r} not in answer_profiles"
            )
        profile = self.answer_profiles[itype]

        if condition is not None:
            polarity = "true" if condition else "false"
            if polarity in self.answer_profiles:
                itype = polarity
                profile = self.answer_profiles[itype]

        instr_i, instr_line = _pick_optional(profile.instruction_snippets)
        ans_i, ans_line = _pick_required(
            profile.answer_templates, label=f"answer_templates[{itype}]"
        )
        answer_line = instr_line
        if answer_line and ans_line:
            answer_line = f"{answer_line} {ans_line}".strip()
        elif ans_line:
            answer_line = ans_line
        answer_index = ans_i
        _ = instr_i

        q_parts = [p for p in (intro_line, stem_line, qinstr_line) if p]
        q_text = " ".join(q_parts)
        a_text = answer_line

        if shared:
            q_text = _fill(q_text, shared)
            a_text = _fill(a_text, shared)
        if q_args:
            q_text = _fill(q_text, q_args)
        if a_args:
            a_text = _fill(a_text, a_args)

        q_bindings: Dict[str, str] = {}
        for line in (intro_line, stem_line, qinstr_line):
            if line:
                q_bindings.update(_bindings_for_line(line, shared, q_args))
        a_bindings = _bindings_for_line(answer_line, shared, a_args)

        return PromptRenderRecord(
            template_id=self.template_id,
            question_index=stem_i,
            answer_index=answer_index,
            question_line=stem_line,
            answer_line=answer_line,
            question_text=q_text.strip(),
            answer_text=a_text.strip(),
            question_bindings=q_bindings,
            answer_bindings=a_bindings,
            question_type=self.question_type.value,
            instruction_type=itype,
            introduction_index=intro_i,
            question_instruction_index=qinstr_i,
        )

    def to_prompt(self, **kwargs) -> str:
        return self.render_provenance(**kwargs).to_prompt()


def legacy_question_type(
    *,
    has_options_placeholder: bool = False,
    has_true_false_answers: bool = False,
    explicit_mcq: bool = False,
) -> QuestionType:
    """Classify legacy flat templates for L2 shim (not used in L1 production path)."""
    if explicit_mcq or has_options_placeholder:
        return QuestionType.MCQ
    if has_true_false_answers:
        return QuestionType.JUDGMENT
    return QuestionType.OPEN_ENDED


def from_legacy_prompt_template(
    template_id: str,
    legacy: PromptTemplate,
    question_type: QuestionType,
    *,
    default_instruction_type: str = "default",
    introduction: Optional[List[str]] = None,
    question_instruction: Optional[List[str]] = None,
    enabled_answer_instruction_types: Optional[List[str]] = None,
) -> StructuredPromptTemplate:
    """
    L2 shim: map flat questions[]/answers[] into structured pools.

    - stem <- questions
    - single answer profile <- answers (or true/false for judgment)
    """
    intro = list(introduction or [])
    qinstr = list(question_instruction or [])
    profiles: Dict[str, AnswerInstructionProfile] = {}

    if question_type == QuestionType.JUDGMENT and legacy.true_answers and legacy.false_answers:
        profiles["true"] = AnswerInstructionProfile(
            "true",
            answer_templates=list(legacy.true_answers),
        )
        profiles["false"] = AnswerInstructionProfile(
            "false",
            answer_templates=list(legacy.false_answers),
        )
        enabled = enabled_answer_instruction_types or ["true", "false"]
    else:
        profiles[default_instruction_type] = AnswerInstructionProfile(
            default_instruction_type,
            answer_templates=list(legacy.answers) if legacy.answers else [""],
        )
        enabled = enabled_answer_instruction_types or [default_instruction_type]

    return StructuredPromptTemplate(
        template_id=template_id,
        question_type=question_type,
        introduction=intro,
        stem=list(legacy.questions),
        question_instruction=qinstr,
        answer_profiles=profiles,
        enabled_answer_instruction_types=enabled,
    )


class StructuredTemplateRegistry:
    _store: Dict[str, StructuredPromptTemplate] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def register(cls, tpl: StructuredPromptTemplate) -> None:
        with cls._lock:
            cls._store[tpl.template_id] = tpl

    @classmethod
    def register_legacy(
        cls,
        template_id: str,
        legacy: PromptTemplate,
        question_type: QuestionType,
        **kwargs,
    ) -> StructuredPromptTemplate:
        tpl = from_legacy_prompt_template(template_id, legacy, question_type, **kwargs)
        cls.register(tpl)
        return tpl

    @classmethod
    def get(cls, template_id: str) -> StructuredPromptTemplate:
        if template_id not in cls._store:
            raise KeyError(
                f"Structured template {template_id!r} not registered. "
                f"Available: {list(cls._store.keys())}"
            )
        return cls._store[template_id]

    @classmethod
    def keys(cls) -> List[str]:
        return list(cls._store.keys())
