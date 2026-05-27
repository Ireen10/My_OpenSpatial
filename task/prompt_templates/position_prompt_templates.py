# position.{height_higher,height_lower,near_far}.{direct|sentence|free}
# Question-side instruction pools only (never answer instruction_snippets).

position_introduction = [
    "Consider the real-world 3D locations of the objects.",
    "Based on the 3D positions of the objects.",
    "Looking at the real-world 3D arrangement.",
    "Considering the spatial layout.",
]

height_higher_stems = [
    "Which object has a higher location? [O]",
    "Which one is placed at a higher elevation? [O]",
    "Which object is positioned higher? [O]",
    "Which one sits higher in 3D space? [O]",
]

height_lower_stems = [
    "Which object has a lower location? [O]",
    "Which one is placed at a lower elevation? [O]",
    "Which object is positioned lower? [O]",
    "Which one sits lower in 3D space? [O]",
]

near_far_stems = [
    "Are the [A] and the [B] close together or far apart? [O]",
    "Would you characterize the spatial proximity of the [A] and the [B] as near or far? [O]",
    "Are the [A] and the [B] near or far relative to one another? [O]",
    "Would you describe the [A] and the [B] as near or far from one another? [O]",
]

position_mcq_sentence_answers = [
    "[P]. Therefore the correct option is [X].",
    "[P]. So the answer is [X].",
    "[P]. The correct option is [X].",
]

position_sentence_instructions = [
    "Answer in a complete sentence.",
    "Reply using a full sentence.",
    "Please describe your answer in a complete sentence.",
]

from ..annotation.core.structured_prompt_template import AnswerInstructionProfile
from .register_structured import (
    EMPTY_QUESTION_INSTRUCTION,
    MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS,
    register_mcq,
)

position_mcq_direct_instructions = MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS

_POSITION_MCQ_MODES = ("direct", "sentence", "free")


def _register_position_mcq_family(template_id: str, stems: list) -> None:
    register_mcq(
        f"{template_id}.direct",
        stems,
        answers=["[X]"],
        introduction=position_introduction,
        question_instruction=position_mcq_direct_instructions,
        answer_profiles={
            "direct": AnswerInstructionProfile("direct", answer_templates=["[X]"]),
        },
        enabled=["direct"],
    )
    register_mcq(
        f"{template_id}.sentence",
        stems,
        answers=position_mcq_sentence_answers,
        introduction=position_introduction,
        question_instruction=position_sentence_instructions,
        answer_profiles={
            "sentence": AnswerInstructionProfile(
                "sentence", answer_templates=position_mcq_sentence_answers
            ),
        },
        enabled=["sentence"],
    )
    register_mcq(
        f"{template_id}.free",
        stems,
        answers=position_mcq_sentence_answers,
        introduction=position_introduction,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
        answer_profiles={
            "free": AnswerInstructionProfile(
                "free", answer_templates=position_mcq_sentence_answers
            ),
        },
        enabled=["free"],
    )


def register_structured_position_templates() -> None:
    for template_id, stems in (
        ("position.height_higher", height_higher_stems),
        ("position.height_lower", height_lower_stems),
        ("position.near_far", near_far_stems),
    ):
        _register_position_mcq_family(template_id, stems)


register_structured_position_templates()
