"""Helpers for registering StructuredPromptTemplate pools (M8 L2)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# Unconstrained ("free") instruction mode: no question_instruction line (empty pool).
EMPTY_QUESTION_INSTRUCTION: List[str] = []

# Shared question-side instruction pools used across tasks.
SENTENCE_QUESTION_INSTRUCTIONS: List[str] = [
    "Answer in a complete sentence.",
    "Reply using a full sentence.",
    "Please describe your answer in a complete sentence.",
]

# Shared introduction pool for multiview tasks.
MULTIVIEW_SCENE_INTRODUCTION: List[str] = [
    "The images show the same scene captured from different viewpoints.",
    "You are viewing multiple perspectives of one shared scene.",
    "These multi-view images depict the same environment from different camera poses.",
    "The provided views are different angles of the same space.",
    "All images represent the same scene under different viewpoints.",
]

# MCQ answer-format hints (mix short + explicit; no concrete option-letter examples in pool).
MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS: List[str] = [
    "Answer with the correct option.",
    "Reply with the chosen option.",
    "Give the option label and option text.",
    "Answer with the selected option label together with the full text of that option.",
    "Reply using both the chosen option identifier and its complete wording.",
    "Give the option label and the matching option text in your answer.",
]

MCQ_ANSWER_WITH_OPTION_AND_NAME_INSTRUCTIONS: List[str] = [
    "Answer with the correct option.",
    "Reply with the chosen option.",
    "Give the option letter and object name.",
    "Answer with the selected option label together with the object named in that option.",
    "Reply using both the chosen option identifier and the corresponding object name.",
    "Give the option label and the named object from that option in your answer.",
]

from ..annotation.core.question_type import QuestionType
from ..annotation.core.structured_prompt_template import (
    AnswerInstructionProfile,
    StructuredPromptTemplate,
    StructuredTemplateRegistry,
)


def default_profile(answers: Sequence[str]) -> Dict[str, AnswerInstructionProfile]:
    return {
        "default": AnswerInstructionProfile(
            "default",
            answer_templates=list(answers),
        ),
    }


def mode_answer_profiles(
    mode: str,
    answers: Sequence[str],
) -> Dict[str, AnswerInstructionProfile]:
    """Single enabled profile keyed by constraint mode (matches template id suffix)."""
    return {
        mode: AnswerInstructionProfile(mode, answer_templates=list(answers)),
    }


def register_oe_mode(
    template_id: str,
    mode: str,
    stem: Sequence[str],
    answers: Sequence[str],
    *,
    introduction: Optional[Sequence[str]] = None,
    question_instruction: Optional[Sequence[str]] = None,
    answer_profiles: Optional[Dict[str, AnswerInstructionProfile]] = None,
) -> None:
    profiles = answer_profiles or mode_answer_profiles(mode, answers)
    if mode not in profiles:
        raise ValueError(f"{template_id}: answer_profiles must include mode {mode!r}")
    register_oe(
        template_id,
        stem,
        answer_profiles=profiles,
        instruction_types=[mode],
        introduction=introduction,
        question_instruction=question_instruction,
    )


def register_mcq_mode(
    template_id: str,
    mode: str,
    stem: Sequence[str],
    *,
    answers: Optional[Sequence[str]] = None,
    letter_only: bool = False,
    introduction: Optional[Sequence[str]] = None,
    question_instruction: Optional[Sequence[str]] = None,
    answer_profiles: Optional[Dict[str, AnswerInstructionProfile]] = None,
) -> None:
    profiles = answer_profiles or mode_answer_profiles(mode, answers or ["[X]"])
    if mode not in profiles:
        raise ValueError(f"{template_id}: answer_profiles must include mode {mode!r}")
    register_mcq(
        template_id,
        stem,
        answer_profiles=profiles,
        enabled=[mode],
        letter_only=letter_only,
        introduction=introduction,
        question_instruction=question_instruction,
    )


def _direct_answers(pool: Sequence[str]) -> List[str]:
    values = list(pool)
    return [values[0]] if values else ["[X]"]


def register_oe_mode_family(
    base_id: str,
    stems: Sequence[str],
    sentence_answers: Sequence[str],
    *,
    direct_instructions: Sequence[str],
    sentence_instructions: Sequence[str] = SENTENCE_QUESTION_INSTRUCTIONS,
    introduction: Optional[Sequence[str]] = None,
    direct_answers: Optional[Sequence[str]] = None,
) -> None:
    register_oe_mode(
        f"{base_id}.direct",
        "direct",
        stems,
        list(direct_answers) if direct_answers is not None else _direct_answers(sentence_answers),
        introduction=introduction,
        question_instruction=direct_instructions,
    )
    register_oe_mode(
        f"{base_id}.sentence",
        "sentence",
        stems,
        sentence_answers,
        introduction=introduction,
        question_instruction=sentence_instructions,
    )
    register_oe_mode(
        f"{base_id}.free",
        "free",
        stems,
        sentence_answers,
        introduction=introduction,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def register_mcq_mode_family(
    base_id: str,
    stems: Sequence[str],
    sentence_answers: Sequence[str],
    *,
    direct_instructions: Sequence[str],
    sentence_instructions: Sequence[str] = SENTENCE_QUESTION_INSTRUCTIONS,
    introduction: Optional[Sequence[str]] = None,
    direct_answers: Optional[Sequence[str]] = None,
) -> None:
    register_mcq_mode(
        f"{base_id}.direct",
        "direct",
        stems,
        answers=list(direct_answers) if direct_answers is not None else _direct_answers(sentence_answers),
        introduction=introduction,
        question_instruction=direct_instructions,
    )
    register_mcq_mode(
        f"{base_id}.sentence",
        "sentence",
        stems,
        answers=sentence_answers,
        introduction=introduction,
        question_instruction=sentence_instructions,
    )
    register_mcq_mode(
        f"{base_id}.free",
        "free",
        stems,
        answers=sentence_answers,
        introduction=introduction,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def mixed_metric_default_profile(
    metric_answers: Sequence[str],
    semantic_answers: Sequence[str],
    *,
    instruction_type: str = "default",
) -> Dict[str, AnswerInstructionProfile]:
    """Metric + semantic lines in one pool; render picks via ``is_metric_depth``."""
    templates = list(metric_answers) + list(semantic_answers)
    flags = [True] * len(metric_answers) + [False] * len(semantic_answers)
    return {
        instruction_type: AnswerInstructionProfile(
            instruction_type,
            answer_templates=templates,
            answer_requires_metric=flags,
        ),
    }


def letter_only_profile(answers: Optional[Sequence[str]] = None) -> Dict[str, AnswerInstructionProfile]:
    pool = list(answers) if answers else ["[X]", "[X]."]
    return {
        "letter_only": AnswerInstructionProfile(
            "letter_only",
            answer_templates=pool,
        ),
    }


def true_false_profiles(
    true_answers: Sequence[str],
    false_answers: Sequence[str],
) -> Dict[str, AnswerInstructionProfile]:
    return {
        "true": AnswerInstructionProfile("true", answer_templates=list(true_answers)),
        "false": AnswerInstructionProfile("false", answer_templates=list(false_answers)),
    }


def register_oe(
    template_id: str,
    stem: Sequence[str],
    answers: Optional[Sequence[str]] = None,
    *,
    introduction: Optional[Sequence[str]] = None,
    question_instruction: Optional[Sequence[str]] = None,
    instruction_types: Optional[List[str]] = None,
    answer_profiles: Optional[Dict[str, AnswerInstructionProfile]] = None,
) -> None:
    if answer_profiles is None:
        if not answers:
            raise ValueError(f"{template_id}: answers or answer_profiles required")
        profiles = default_profile(answers)
    else:
        profiles = answer_profiles
    enabled = instruction_types or list(profiles.keys())
    StructuredTemplateRegistry.register(
        StructuredPromptTemplate(
            template_id=template_id,
            question_type=QuestionType.OPEN_ENDED,
            introduction=list(introduction or []),
            stem=list(stem),
            question_instruction=list(question_instruction or []),
            answer_profiles=profiles,
            enabled_answer_instruction_types=enabled,
        )
    )


def register_mcq(
    template_id: str,
    stem: Sequence[str],
    *,
    answers: Optional[Sequence[str]] = None,
    letter_only: bool = True,
    introduction: Optional[Sequence[str]] = None,
    question_instruction: Optional[Sequence[str]] = None,
    answer_profiles: Optional[Dict[str, AnswerInstructionProfile]] = None,
    enabled: Optional[List[str]] = None,
) -> None:
    if answer_profiles is None:
        answer_profiles = (
            letter_only_profile(answers)
            if letter_only
            else default_profile(answers or ["[X]"])
        )
    StructuredTemplateRegistry.register(
        StructuredPromptTemplate(
            template_id=template_id,
            question_type=QuestionType.MCQ,
            introduction=list(introduction or []),
            stem=list(stem),
            question_instruction=list(question_instruction or []),
            answer_profiles=answer_profiles,
            enabled_answer_instruction_types=enabled or list(answer_profiles.keys()),
        )
    )


def register_judgment(
    template_id: str,
    stem: Sequence[str],
    true_answers: Sequence[str],
    false_answers: Sequence[str],
    *,
    introduction: Optional[Sequence[str]] = None,
    question_instruction: Optional[Sequence[str]] = None,
) -> None:
    profiles = true_false_profiles(true_answers, false_answers)
    StructuredTemplateRegistry.register(
        StructuredPromptTemplate(
            template_id=template_id,
            question_type=QuestionType.JUDGMENT,
            introduction=list(introduction or []),
            stem=list(stem),
            question_instruction=list(question_instruction or []),
            answer_profiles=profiles,
            enabled_answer_instruction_types=["true", "false"],
        )
    )
