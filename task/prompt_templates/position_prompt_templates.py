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

from .register_structured import SENTENCE_QUESTION_INSTRUCTIONS
from .register_structured import (
    MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS,
    register_mcq_mode_family,
)

position_mcq_direct_instructions = MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS

def _register_position_mcq_family(template_id: str, stems: list) -> None:
    register_mcq_mode_family(
        template_id,
        stems,
        position_mcq_sentence_answers,
        direct_answers=["[X]"],
        direct_instructions=position_mcq_direct_instructions,
        sentence_instructions=SENTENCE_QUESTION_INSTRUCTIONS,
        introduction=position_introduction,
    )


def register_structured_position_templates() -> None:
    for template_id, stems in (
        ("position.height_higher", height_higher_stems),
        ("position.height_lower", height_lower_stems),
        ("position.near_far", near_far_stems),
    ):
        _register_position_mcq_family(template_id, stems)


register_structured_position_templates()
