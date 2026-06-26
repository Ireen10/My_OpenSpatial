# Template IDs:
#   distance.absolute_*           — singleview only (no introduction)
#   multiview_distance.absolute_* — same stems as distance.absolute_* + introduction
#   multiview_distance.*          — multiview-only (farthest/closest, obj_cam)
#
from .register_structured import (
    EMPTY_QUESTION_INSTRUCTION,
    MCQ_ANSWER_WITH_OPTION_AND_NAME_INSTRUCTIONS,
    MULTIVIEW_SCENE_INTRODUCTION,
    SENTENCE_QUESTION_INSTRUCTIONS,
    mixed_metric_default_profile,
    register_mcq,
    register_mcq_mode,
    register_oe_mode,
    register_oe_mode_family,
)

# ─── Shared: absolute distance (OE) ─────────────────────────────────────

absolute_distance_stems = [
    "Measuring from the closest point of each object, what is the distance between the [A] and the [B]?",
    "What is the distance between the [A] and the [B]?",
    "Consider the real-world 3D location of the objects. What is the distance between the [A] and the [B]?",
]

absolute_direct_instructions = [
    "Give the numeric distance with the appropriate unit (meters or centimeters).",
    "Reply with the distance using meters or centimeters as appropriate.",
    "State their separation as a single numeric value with unit.",
]

absolute_sentence_instructions = [
    "Answer in a complete sentence that includes the distance and unit.",
    "Reply using a full sentence with the distance and appropriate unit.",
    "Provide your final answer in a complete sentence including the distance.",
]

absolute_direct_answers = ["[X]"]
absolute_sentence_answers = [
    "The distance between the [A] and the [B] is [X].",
    "The [A] and the [B] are approximately [X] apart.",
]

# ─── Shared: singleview relative distance (OE / MCQ); multiview N-ary uses multiview_distance.* ──

positional_far_oe_questions = [
    "Estimate the real-world distances between objects in this image. Which object is farther from the [C], the [A] or the [B]?",
    "Based on the spatial arrangement of objects in this image, which object is more distant from the [C], the [A] or the [B]?",
    "Considering the 3D positions of objects in this image, which one is farther from the [C], the [A] or the [B]?",
    "From the perspective of this image, which object is more distant from the [C], the [A] or the [B]?",
    "Looking at the spatial layout in this image, which object is farther from the [C], the [A] or the [B]?",
    "Which of the [A] and the [B] is farther from the [C]?",
]

positional_far_mcq_questions = [
    "Estimate the real-world distances between objects in this image. Which object is farther from the [C], the [A] or the [B]? [O]",
    "Based on the spatial arrangement of objects in this image, which object is more distant from the [C], the [A] or the [B]? [O]",
    "Considering the 3D positions of objects in this image, which one is farther from the [C], the [A] or the [B]? [O]",
    "From the perspective of this image, which object is more distant from the [C], the [A] or the [B]? [O]",
    "Looking at the spatial layout in this image, which object is farther from the [C], the [A] or the [B]? [O]",
    "Which of the [A] and the [B] is farther from the [C]? [O]",
]

positional_far_oe_direct_answers = [
    "[X]."
]

positional_far_oe_reasoning_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. Therefore, [X] is farther from [C].",
    "Given these two distances, [X] is farther from [C].",
]
positional_far_oe_reasoning_semantic_answers = [
    "The [X] is farther from the [C].",
    "The [X] is farther from the [C] than the [G] is.",
]

positional_far_mcq_direct_answers = ["[X]", "[X]."]
positional_far_mcq_reasoning_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. Therefore, the correct option is [X].",
]
positional_far_mcq_reasoning_semantic_answers = [
    "The [F] is farther from the [C] than the [G] is. Therefore, the correct option is [X].",
    "The [F] is farther from the [C] than the [G] is, so the answer is [X].",
]
positional_close_oe_questions = [
    "Estimate the real-world distances between objects in this image. Which object is closer to the [C], the [A] or the [B]?",
    "Based on the spatial arrangement of objects in this image, which object is nearer to the [C], the [A] or the [B]?",
    "Considering the 3D positions of objects in this image, which one is closer to the [C], the [A] or the [B]?",
    "From the perspective of this image, which object is nearer to the [C], the [A] or the [B]?",
    "Looking at the spatial layout in this image, which object is closer to the [C], the [A] or the [B]?",
    "Which of the [A] and the [B] is closer to the [C]?",
]

positional_close_mcq_questions = [
    "Estimate the real-world distances between objects in this image. Which object is closer to the [C], the [A] or the [B]? [O]",
    "Based on the spatial arrangement of objects in this image, which object is nearer to the [C], the [A] or the [B]? [O]",
    "Considering the 3D positions of objects in this image, which one is closer to the [C], the [A] or the [B]? [O]",
    "From the perspective of this image, which object is nearer to the [C], the [A] or the [B]? [O]",
    "Looking at the spatial layout in this image, which object is closer to the [C], the [A] or the [B]? [O]",
    "Which of the [A] and the [B] is closer to the [C]? [O]",
]

positional_close_oe_direct_answers = [
    "[X]."
]

positional_close_oe_reasoning_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. Therefore, [X] is closer to [C].",
    "Given these two distances, [X] is closer to [C].",
]
positional_close_oe_reasoning_semantic_answers = [
    "The [X] is closer to the [C].",
    "The [X] is closer to the [C] than the [G] is.",
]

positional_close_mcq_direct_answers = ["[X]."]
positional_close_mcq_reasoning_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. Therefore, the correct option is [X].",
]
positional_close_mcq_reasoning_semantic_answers = [
    "The [F] is closer to the [C] than the [G] is. Therefore, the correct option is [X].",
    "The [F] is closer to the [C] than the [G] is, so the answer is [X].",
]
relative_oe_direct_instructions = [
    "Answer with the object name only.",
    "Give the correct object as your direct answer.",
    "Reply using only the object name.",
]

relative_oe_reasoning_instructions = [
    "Estimate each distance to the anchor, then compare.",
    "First estimate the two absolute distances to the anchor, then answer.",
    "Reason from the two absolute distances, then state which object applies.",
]

relative_oe_sentence_instructions = SENTENCE_QUESTION_INSTRUCTIONS

relative_mcq_sentence_instructions = [
    "Answer in a complete sentence.",
    "Reply using a full sentence.",
    "Answer in a complete sentence that states the correct option.",
]

positional_far_oe_sentence_answers = [
    "The [X] is farther from the [C].",
    "The [X] is farther from the [C] than the [G] is.",
]

positional_close_oe_sentence_answers = [
    "The [X] is closer to the [C].",
    "The [X] is closer to the [C] than the [G] is.",
]

positional_far_mcq_sentence_answers = [
    "[P]. Therefore the correct option is [X].",
    "[P]. So the answer is [X].",
    "[P]. The correct option is [X].",
]

positional_close_mcq_sentence_answers = [
    "[P]. Therefore the correct option is [X].",
    "[P]. So the answer is [X].",
    "[P]. The correct option is [X].",
]

RELATIVE_POLARITY_POOLS = {
    "far": {
        "oe_stems": positional_far_oe_questions,
        "oe_direct_answers": positional_far_oe_direct_answers,
        "oe_sentence_answers": positional_far_oe_sentence_answers,
        "mcq_stems": positional_far_mcq_questions,
        "mcq_direct_answers": positional_far_mcq_direct_answers,
        "mcq_sentence_answers": positional_far_mcq_sentence_answers,
        "oe_reasoning_metric_answers": positional_far_oe_reasoning_metric_answers,
        "oe_reasoning_semantic_answers": positional_far_oe_reasoning_semantic_answers,
        "mcq_reasoning_metric_answers": positional_far_mcq_reasoning_metric_answers,
        "mcq_reasoning_semantic_answers": positional_far_mcq_reasoning_semantic_answers,
    },
    "close": {
        "oe_stems": positional_close_oe_questions,
        "oe_direct_answers": positional_close_oe_direct_answers,
        "oe_sentence_answers": positional_close_oe_sentence_answers,
        "mcq_stems": positional_close_mcq_questions,
        "mcq_direct_answers": positional_close_mcq_direct_answers,
        "mcq_sentence_answers": positional_close_mcq_sentence_answers,
        "oe_reasoning_metric_answers": positional_close_oe_reasoning_metric_answers,
        "oe_reasoning_semantic_answers": positional_close_oe_reasoning_semantic_answers,
        "mcq_reasoning_metric_answers": positional_close_mcq_reasoning_metric_answers,
        "mcq_reasoning_semantic_answers": positional_close_mcq_reasoning_semantic_answers,
    },
}

relative_mcq_direct_instructions = MCQ_ANSWER_WITH_OPTION_AND_NAME_INSTRUCTIONS

relative_mcq_reasoning_instructions = [
    "Compare distances, then choose the correct option.",
    "Answer with the correct option after comparing distances.",
    "Estimate each distance to the anchor, then compare and choose the correct option.",
    "Reason from the two absolute distances, then answer with the option label and object name.",
    "Compare distances first, then reply with the chosen option identifier and its name.",
]

# ─── Multiview distance (N-ary farthest/closest; obj_cam in multiview_distance_obj_cam.py) ──

multiview_distance_introduction = MULTIVIEW_SCENE_INTRODUCTION

multiview_distance_farthest_questions = [
    "Among the listed objects ([T]), which one is the farthest from the [X]?",
    "Considering the set of objects ([T]), which object is most distant from the [X]?",
    "From the provided objects ([T]), identify the one that is farthest from the [X].",
    "Which object in the list ([T]) has the greatest distance from the [X]?",
    "Out of the listed objects ([T]), which one is the most distant from the [X]?",
    "Which object in the list ([T]) has the maximum distance to the [X]?",
]

multiview_distance_farthest_direct_answers = [
    "[X].",
]

multiview_distance_farthest_sentence_answers = [
    "The [X] is farther from the [Y] than any other listed object.",
    "Among the listed objects, the [X] has the greatest distance to the [Y].",
    "Compared with the other listed objects, the [X] is the most distant from the [Y].",
]

multiview_distance_farthest_reasoning_answers = [
    "After comparing the listed objects by their distance from the [Y], the [X] is the most distant one. Therefore, the [X] is the farthest from the [Y].",
    "The object farthest from the [Y] is the one with the greatest separation among the candidates. That object is the [X].",
    "Comparing the [X] with the other listed objects, the [X] has the largest distance to the [Y], so it is the farthest.",
]

multiview_distance_closest_questions = [
    "Among the listed objects ([T]), which one is the closest to the [X]?",
    "Considering the set of objects ([T]), which object is nearest to the [X]?",
    "From the provided objects ([T]), identify the one that is closest to the [X].",
    "Which object in the list ([T]) has the smallest distance from the [X]?",
    "Out of the listed objects ([T]), which one is the nearest to the [X]?",
    "Which object in the list ([T]) has the minimum distance to the [X]?",
]

multiview_distance_closest_direct_answers = [
    "[X].",
]

multiview_distance_closest_sentence_answers = [
    "The [X] is closer to the [Y] than any other listed object.",
    "Among the listed objects, the [X] has the smallest distance to the [Y].",
    "Compared with the other listed objects, the [X] is the nearest to the [Y].",
]

multiview_distance_closest_reasoning_answers = [
    "After comparing the listed objects by their distance from the [Y], the [X] is the nearest one. Therefore, the [X] is the closest to the [Y].",
    "The object closest to the [Y] is the one with the smallest separation among the candidates. That object is the [X].",
    "Comparing the [X] with the other listed objects, the [X] has the smallest distance to the [Y], so it is the closest.",
]

multiview_distance_obj_cam_questions = [
    "In which view is the [A] [Y] to the spot where the camera was positioned?",
    "Which view shows the [A] [Y] to the camera position?",
    "In which view does the [A] appear [Y] to where the camera was placed?",
    "Which view is the [A] [Y] to the camera viewpoint?",
    "Between View 1 and View 2, in which view is the [A] [Y] to the camera?",
]

multiview_distance_obj_cam_option_phrases = [
    "[Y] to the spot where camera [X] was positioned",
    "The [A] is [Y] to the camera in [X].",
    "In [X], the [A] is [Y] to the camera position.",
]
# Backward-compatible alias used by annotation module imports.
multiview_distance_obj_cam_answers = multiview_distance_obj_cam_option_phrases
multiview_distance_obj_cam_equal_option = (
    "distance to the spot where camera View 1 and View 2 were positioned is equal"
)

multiview_distance_obj_cam_mcq_questions = [q + "\n[O]" for q in multiview_distance_obj_cam_questions]

multiview_distance_obj_cam_mcq_answers = [
    "[X].",
]
def _relative_oe_reasoning_profiles(polarity: str):
    pools = RELATIVE_POLARITY_POOLS[polarity]
    return mixed_metric_default_profile(
        pools["oe_reasoning_metric_answers"],
        pools["oe_reasoning_semantic_answers"],
        instruction_type="reasoning",
    )


def _relative_mcq_reasoning_profiles(polarity: str):
    pools = RELATIVE_POLARITY_POOLS[polarity]
    return mixed_metric_default_profile(
        pools["mcq_reasoning_metric_answers"],
        pools["mcq_reasoning_semantic_answers"],
        instruction_type="reasoning",
    )


def _register_relative_oe_pair(polarity: str) -> None:
    """Register far/close OE: direct / reasoning / sentence / free."""
    pools = RELATIVE_POLARITY_POOLS[polarity]
    stems = pools["oe_stems"]
    direct_ans = pools["oe_direct_answers"]
    sentence_ans = pools["oe_sentence_answers"]

    reasoning_profiles = _relative_oe_reasoning_profiles(polarity)

    register_oe_mode(
        f"distance.relative_{polarity}.direct",
        "direct",
        stems,
        direct_ans,
        question_instruction=relative_oe_direct_instructions,
    )
    register_oe_mode(
        f"distance.relative_{polarity}.reasoning",
        "reasoning",
        stems,
        [],
        answer_profiles=reasoning_profiles,
        question_instruction=relative_oe_reasoning_instructions,
    )
    register_oe_mode(
        f"distance.relative_{polarity}.sentence",
        "sentence",
        stems,
        sentence_ans,
        question_instruction=relative_oe_sentence_instructions,
    )
    register_oe_mode(
        f"distance.relative_{polarity}.free",
        "free",
        stems,
        sentence_ans,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def _register_relative_mcq_pair(polarity: str) -> None:
    pools = RELATIVE_POLARITY_POOLS[polarity]
    stems = pools["mcq_stems"]
    direct_ans = pools["mcq_direct_answers"]
    sentence_ans = pools["mcq_sentence_answers"]

    reasoning_profiles = _relative_mcq_reasoning_profiles(polarity)

    register_mcq_mode(
        f"distance.relative_{polarity}_mcq.direct",
        "direct",
        stems,
        answers=direct_ans,
        question_instruction=relative_mcq_direct_instructions,
    )
    register_mcq_mode(
        f"distance.relative_{polarity}_mcq.reasoning",
        "reasoning",
        stems,
        answer_profiles=reasoning_profiles,
        question_instruction=relative_mcq_reasoning_instructions,
    )
    register_mcq_mode(
        f"distance.relative_{polarity}_mcq.sentence",
        "sentence",
        stems,
        answers=sentence_ans,
        question_instruction=relative_mcq_sentence_instructions,
    )
    register_mcq_mode(
        f"distance.relative_{polarity}_mcq.free",
        "free",
        stems,
        answers=sentence_ans,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def _register_absolute_distance_family(prefix: str, *, introduction=None) -> None:
    register_oe_mode_family(
        prefix,
        absolute_distance_stems,
        absolute_sentence_answers,
        direct_answers=absolute_direct_answers,
        direct_instructions=absolute_direct_instructions,
        sentence_instructions=absolute_sentence_instructions,
        introduction=introduction,
    )


def _register_multiview_superlative_family(
    polarity: str,
    stems: list,
    direct_answers: list,
    sentence_answers: list,
    reasoning_answers: list,
) -> None:
    register_oe_mode(
        f"multiview_distance.{polarity}.direct",
        "direct",
        stems,
        direct_answers,
        introduction=multiview_distance_introduction,
        question_instruction=relative_oe_direct_instructions,
    )
    register_oe_mode(
        f"multiview_distance.{polarity}.reasoning",
        "reasoning",
        stems,
        reasoning_answers,
        introduction=multiview_distance_introduction,
        question_instruction=relative_oe_reasoning_instructions,
    )
    register_oe_mode(
        f"multiview_distance.{polarity}.sentence",
        "sentence",
        stems,
        sentence_answers,
        introduction=multiview_distance_introduction,
        question_instruction=relative_oe_sentence_instructions,
    )
    register_oe_mode(
        f"multiview_distance.{polarity}.free",
        "free",
        stems,
        [],
        answer_profiles=mixed_metric_default_profile(
            reasoning_answers,
            sentence_answers,
            instruction_type="free",
        ),
        introduction=multiview_distance_introduction,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def register_structured_distance_templates() -> None:
    _register_absolute_distance_family("distance.absolute")
    _register_absolute_distance_family(
        "multiview_distance.absolute",
        introduction=multiview_distance_introduction,
    )

    _register_relative_oe_pair("far")
    _register_relative_oe_pair("close")
    _register_relative_mcq_pair("far")
    _register_relative_mcq_pair("close")

    for polarity, stems, direct_ans, sentence_ans, reasoning_ans in (
        (
            "farthest",
            multiview_distance_farthest_questions,
            multiview_distance_farthest_direct_answers,
            multiview_distance_farthest_sentence_answers,
            multiview_distance_farthest_reasoning_answers,
        ),
        (
            "closest",
            multiview_distance_closest_questions,
            multiview_distance_closest_direct_answers,
            multiview_distance_closest_sentence_answers,
            multiview_distance_closest_reasoning_answers,
        ),
    ):
        _register_multiview_superlative_family(
            polarity,
            stems,
            direct_ans,
            sentence_ans,
            reasoning_ans,
        )
    register_mcq(
        "multiview_distance.obj_cam_mcq",
        multiview_distance_obj_cam_mcq_questions,
        answers=multiview_distance_obj_cam_mcq_answers,
        letter_only=False,
        introduction=multiview_distance_introduction,
    )


register_structured_distance_templates()
