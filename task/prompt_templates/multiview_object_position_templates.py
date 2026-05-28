# multiview_position.{object_relative,viewer_at_anchor}[._mcq].{direct|sentence|free}
# Instruction constraints on the question only (question_instruction pools).

object_relative_stems = [
    "If the [A] is [X] of the [B] in image 1, what direction is the [C] (visible in image 2) from the [B]?",
    "If the [A] is to the [X] of the [B] in the first image, what direction is the [C] from the [B]?",
    "Given that the [A] appears [X] relative to the [B] in image 1, which direction does the [C] (seen in image 2) lie with respect to the [B]?",
    "In image 1, if the [A] is located [X] of the [B], what direction does the [C] (depicted in image 2) take from the [B]?",
    "If the [A] is positioned [X] relative to the [B] in the first image, how would you describe the direction of the [C] (visible in image 2) in relation to the [B]?",
    "What direction does the [C] (shown in image 2) occupy from the [B], given that the [A] is [X] to the [B] in image 1?",
]

object_relative_premises = [
    "If the [A] is [X] of the [B] in image 1",
    "If the [A] is to the [X] of the [B] in the first image",
    "Given that the [A] appears [X] relative to the [B] in image 1",
    "In image 1, if the [A] is located [X] of the [B]",
    "If the [A] is positioned [X] relative to the [B] in the first image",
    "Given that the [A] is [X] to the [B] in image 1",
]

viewer_at_anchor_stems = [
    "If I am at the position of the [B] in image 1, and the [A] is on the [X] side of me, what direction is the [C] (visible in image 2) from my position?",
    "Standing at the location of the [B] in the first image, with the [A] on my [X] side, which direction does the [C] (seen in image 2) lie from me?",
    "From the viewpoint of the [B] in image 1, if the [A] is located at the [X] side of me, what direction does the [C] (depicted in image 2) take from my position?",
    "If I consider myself at the [B]'s position in the first image, and the [A] is positioned at the [X] side of me, how would I describe the direction of the [C] (visible in image 2) from my location?",
    "Assume I am at the [B]'s position in image 1, with the [A] on my [X] side, what direction does the [C] (shown in image 2) occupy from my viewpoint?",
    "From the perspective of the [B] in the first image, if the [A] is on the [X] side of the [B], which direction is the [C] (visible in image 2) from the [B]'s position?",
]

viewer_at_anchor_premises = [
    "If I am at the position of the [B] in image 1, and the [A] is on the [X] side of me",
    "Standing at the location of the [B] in the first image, with the [A] on my [X] side",
    "From the viewpoint of the [B] in image 1, if the [A] is located at the [X] side of me",
    "If I consider myself at the [B]'s position in the first image, and the [A] is positioned at the [X] side of me",
    "Assume I am at the [B]'s position in image 1, with the [A] on my [X] side",
    "From the perspective of the [B] in the first image, if the [A] is on the [X] side of the [B]",
]

FRAME_PREMISE_POOLS = {
    "object_relative": object_relative_premises,
    "viewer_at_anchor": viewer_at_anchor_premises,
}

object_relative_stems_mcq = [q + "\n[O]" for q in object_relative_stems]
viewer_at_anchor_stems_mcq = [q + "\n[O]" for q in viewer_at_anchor_stems]

object_relative_oe_sentence_answers = [
    "[P], the [C] is [D] of the [B].",
    "[P], the [C] lies to the [D] of the [B].",
    "[P], the [C] is located to the [D] of the [B].",
]

viewer_at_anchor_oe_sentence_answers = [
    "[P], the [C] is on the [D] side.",
    "[P], from my position at the [B], the [C] is on the [D] side.",
    "[P], the [C] is on the [D] side relative to me.",
]

object_relative_mcq_sentence_answers = [
    "[P], the [C] is [D] of the [B]. Therefore the correct option is [E].",
    "[P], the [C] lies to the [D] of the [B]. So the answer is [E].",
    "[P], the [C] is [D] of the [B]. The correct option is [E].",
]

viewer_at_anchor_mcq_sentence_answers = [
    "[P], the [C] is on the [D] side. Therefore the correct option is [E].",
    "[P], from my position at the [B], the [C] is on the [D] side. So the answer is [E].",
    "[P], the [C] is on the [D] side relative to me. The correct option is [E].",
]

from .register_structured import (
    EMPTY_QUESTION_INSTRUCTION,
    MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS,
    MULTIVIEW_SCENE_INTRODUCTION,
    SENTENCE_QUESTION_INSTRUCTIONS,
    register_mcq_mode,
    register_oe_mode,
)

multiview_position_introduction = MULTIVIEW_SCENE_INTRODUCTION
position_sentence_instructions = SENTENCE_QUESTION_INSTRUCTIONS
position_mcq_direct_instructions = MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS

def _register_oe_family(base_id: str, stems: list, sentence_answers: list) -> None:
    register_oe_mode(
        f"{base_id}.sentence",
        "sentence",
        stems,
        sentence_answers,
        introduction=multiview_position_introduction,
        question_instruction=position_sentence_instructions,
    )
    register_oe_mode(
        f"{base_id}.free",
        "free",
        stems,
        sentence_answers,
        introduction=multiview_position_introduction,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def _register_mcq_family(base_id: str, stems: list, sentence_answers: list) -> None:
    register_mcq_mode(
        f"{base_id}.direct",
        "direct",
        stems,
        answers=["[T]"],
        introduction=multiview_position_introduction,
        question_instruction=position_mcq_direct_instructions,
    )
    register_mcq_mode(
        f"{base_id}.sentence",
        "sentence",
        stems,
        answers=sentence_answers,
        introduction=multiview_position_introduction,
        question_instruction=position_sentence_instructions,
    )
    register_mcq_mode(
        f"{base_id}.free",
        "free",
        stems,
        answers=sentence_answers,
        introduction=multiview_position_introduction,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def register_structured_multiview_position_templates() -> None:
    _register_oe_family(
        "multiview_position.object_relative",
        object_relative_stems,
        object_relative_oe_sentence_answers,
    )
    _register_mcq_family(
        "multiview_position.object_relative_mcq",
        object_relative_stems_mcq,
        object_relative_mcq_sentence_answers,
    )
    _register_oe_family(
        "multiview_position.viewer_at_anchor",
        viewer_at_anchor_stems,
        viewer_at_anchor_oe_sentence_answers,
    )
    _register_mcq_family(
        "multiview_position.viewer_at_anchor_mcq",
        viewer_at_anchor_stems_mcq,
        viewer_at_anchor_mcq_sentence_answers,
    )


register_structured_multiview_position_templates()
