"""Helpers for registering StructuredPromptTemplate pools (M8 L2)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# Unconstrained ("free") instruction mode: no question_instruction line (empty pool).
EMPTY_QUESTION_INSTRUCTION: List[str] = []

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
    answers: Sequence[str],
    *,
    introduction: Optional[Sequence[str]] = None,
    question_instruction: Optional[Sequence[str]] = None,
    instruction_types: Optional[List[str]] = None,
    answer_profiles: Optional[Dict[str, AnswerInstructionProfile]] = None,
) -> None:
    profiles = answer_profiles or default_profile(answers)
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
