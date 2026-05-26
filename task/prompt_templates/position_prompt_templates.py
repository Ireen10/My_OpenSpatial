# Template IDs: position.height_higher | height_lower | near_far (MCQ, direct answer only)

position_introduction = [
    "Consider the real-world 3D locations of the objects.",
    "Based on the 3D positions of the objects.",
    "Looking at the real-world 3D arrangement.",
    "Considering the spatial layout.",
]

from .register_structured import MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS

position_mcq_direct_instructions = MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS

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

mcq_answers = ["[X]"]

from .register_structured import register_mcq


def register_structured_position_templates() -> None:
    for template_id, stems in (
        ("position.height_higher", height_higher_stems),
        ("position.height_lower", height_lower_stems),
        ("position.near_far", near_far_stems),
    ):
        register_mcq(
            template_id,
            stems,
            answers=mcq_answers,
            introduction=position_introduction,
            question_instruction=position_mcq_direct_instructions,
        )


register_structured_position_templates()
