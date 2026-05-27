# Template IDs:
#   distance.absolute_*           — singleview only (no introduction)
#   multiview_distance.absolute_* — same stems as distance.absolute_* + introduction
#   multiview_distance.*          — multiview-only (farthest/closest, obj_cam)
#
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
    "Which of [A] and [B] is farther to [C]?",
]

positional_far_mcq_questions = [
    "Estimate the real-world distances between objects in this image. Which object is farther from the [C], the [A] or the [B]? [O]",
    "Based on the spatial arrangement of objects in this image, which object is more distant from the [C], the [A] or the [B]? [O]",
    "Considering the 3D positions of objects in this image, which one is farther from the [C], the [A] or the [B]? [O]",
    "From the perspective of this image, which object is more distant from the [C], the [A] or the [B]? [O]",
    "Looking at the spatial layout in this image, which object is farther from the [C], the [A] or the [B]? [O]",
    "Which of [A] and [B] is farther to [C]? [O]",
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
positional_far_mcq_free_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. Therefore, the answer is [X].",
]
positional_far_mcq_free_semantic_answers = [
    "The [F] is farther from the [C] than the [G] is. Therefore, the answer is [X].",
    "The [F] is farther from the [C] than the [G] is, so the answer is [X].",
]

positional_close_oe_questions = [
    "Estimate the real-world distances between objects in this image. Which object is closer to the [C], the [A] or the [B]?",
    "Based on the spatial arrangement of objects in this image, which object is nearer to the [C], the [A] or the [B]?",
    "Considering the 3D positions of objects in this image, which one is closer to the [C], the [A] or the [B]?",
    "From the perspective of this image, which object is nearer to the [C], the [A] or the [B]?",
    "Looking at the spatial layout in this image, which object is closer to the [C], the [A] or the [B]?",
    "Which of [A] and [B] is closer to [C]?",
]

positional_close_mcq_questions = [
    "Estimate the real-world distances between objects in this image. Which object is closer to the [C], the [A] or the [B]? [O]",
    "Based on the spatial arrangement of objects in this image, which object is nearer to the [C], the [A] or the [B]? [O]",
    "Considering the 3D positions of objects in this image, which one is closer to the [C], the [A] or the [B]? [O]",
    "From the perspective of this image, which object is nearer to the [C], the [A] or the [B]? [O]",
    "Looking at the spatial layout in this image, which object is closer to the [C], the [A] or the [B]? [O]",
    "Which of [A] and [B] is closer to [C]? [O]",
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
positional_close_mcq_free_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. Therefore, the answer is [X].",
]
positional_close_mcq_free_semantic_answers = [
    "The [F] is closer to the [C] than the [G] is. Therefore, the answer is [X].",
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

relative_oe_sentence_instructions = [
    "Answer in a complete sentence.",
    "Reply using a full sentence.",
    "Please describe your answer in a complete sentence.",
]

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

from .register_structured import MCQ_ANSWER_WITH_OPTION_AND_NAME_INSTRUCTIONS

relative_mcq_direct_instructions = MCQ_ANSWER_WITH_OPTION_AND_NAME_INSTRUCTIONS

relative_mcq_reasoning_instructions = [
    "Compare distances, then choose the correct option.",
    "Answer with the correct option after comparing distances.",
    "Estimate each distance to the anchor, then compare and choose the correct option.",
    "Reason from the two absolute distances, then answer with the option label and object name.",
    "Compare distances first, then reply with the chosen option identifier and its name.",
]

# ─── Multiview distance (N-ary farthest/closest; obj_cam in multiview_distance_obj_cam.py) ──

multiview_distance_introduction = [
    "The images show the same scene captured from different viewpoints.",
    "You are viewing multiple perspectives of one shared scene.",
    "These multi-view images depict the same environment from different camera poses.",
    "The provided views are different angles of the same space.",
    "All images represent the same scene under different viewpoints.",
]

multiview_distance_farthest_questions = [
    "Among the objects [T], which one is the farthest from [X]?",
    "Considering the set of objects [T], which object is most distant from [X]?",
    "From the provided objects [T], identify the one that is farthest from [X].",
    "Which object in [T] has the greatest distance from [X]?",
    "Out of the objects [T], which one is the most distant from [X]?",
    "Which object in [T] has the maximum distance to [X]?",
]

multiview_distance_farthest_direct_answers = [
    "[X].",
]

multiview_distance_farthest_sentence_answers = [
    "The object [X] is farther from [Y] than any other listed object.",
    "Among the listed objects, [X] has the greatest distance to [Y].",
    "Compared with the other listed objects, [X] is the most distant from [Y].",
]

multiview_distance_closest_questions = [
    "Among the objects [T], which one is the closest to [X]?",
    "Considering the set of objects [T], which object is nearest to [X]?",
    "From the provided objects [T], identify the one that is closest to [X].",
    "Which object in [T] has the smallest distance from [X]?",
    "Out of the objects [T], which one is the nearest to [X]?",
    "Which object in [T] has the minimum distance to [X]?",
]

multiview_distance_closest_direct_answers = [
    "[X].",
]

multiview_distance_closest_sentence_answers = [
    "The object [X] is closer to [Y] than any other listed object.",
    "Among the listed objects, [X] has the smallest distance to [Y].",
    "Compared with the other listed objects, [X] is the nearest to [Y].",
]

multiview_distance_obj_cam_questions = [
    "In which view is the [A] [Y] to the spot where the camera was positioned?",
    "Which view shows the [A] [Y] to the camera position?",
    "In which view does the [A] appear [Y] to where the camera was placed?",
    "Which view is the [A] [Y] to the camera viewpoint?",
    "Between View 1 and View 2, in which view is the [A] [Y] to the camera?",
]

multiview_distance_obj_cam_answers = [
    "[Y] to the spot where camera [X] was positioned",
    "The [A] is [Y] to the camera in [X].",
    "In [X], the [A] is [Y] to the camera position.",
]

multiview_distance_obj_cam_mcq_questions = [q + "\n[O]" for q in multiview_distance_obj_cam_questions]

multiview_distance_obj_cam_mcq_answers = [
    "[X].",
]
from .register_structured import (
    EMPTY_QUESTION_INSTRUCTION,
    mixed_metric_default_profile,
    register_mcq,
    register_mcq_mode,
    register_oe,
    register_oe_mode,
)


def _relative_oe_reasoning_profiles(polarity: str):
    if polarity == "far":
        return mixed_metric_default_profile(
            positional_far_oe_reasoning_metric_answers,
            positional_far_oe_reasoning_semantic_answers,
            instruction_type="reasoning",
        )
    return mixed_metric_default_profile(
        positional_close_oe_reasoning_metric_answers,
        positional_close_oe_reasoning_semantic_answers,
        instruction_type="reasoning",
    )


def _relative_mcq_reasoning_profiles(polarity: str):
    if polarity == "far":
        return mixed_metric_default_profile(
            positional_far_mcq_reasoning_metric_answers,
            positional_far_mcq_reasoning_semantic_answers,
            instruction_type="reasoning",
        )
    return mixed_metric_default_profile(
        positional_close_mcq_reasoning_metric_answers,
        positional_close_mcq_reasoning_semantic_answers,
        instruction_type="reasoning",
    )


def _relative_mcq_free_profiles(polarity: str):
    if polarity == "far":
        return mixed_metric_default_profile(
            positional_far_mcq_free_metric_answers,
            positional_far_mcq_free_semantic_answers,
        )
    return mixed_metric_default_profile(
        positional_close_mcq_free_metric_answers,
        positional_close_mcq_free_semantic_answers,
    )


def _register_relative_oe_pair(polarity: str) -> None:
    """Register far/close OE: direct / reasoning / sentence / free."""
    if polarity == "far":
        stems = positional_far_oe_questions
        direct_ans = positional_far_oe_direct_answers
        sentence_ans = positional_far_oe_sentence_answers
    else:
        stems = positional_close_oe_questions
        direct_ans = positional_close_oe_direct_answers
        sentence_ans = positional_close_oe_sentence_answers

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
    if polarity == "far":
        stems = positional_far_mcq_questions
        direct_ans = positional_far_mcq_direct_answers
        sentence_ans = positional_far_mcq_sentence_answers
    else:
        stems = positional_close_mcq_questions
        direct_ans = positional_close_mcq_direct_answers
        sentence_ans = positional_close_mcq_sentence_answers

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
    kwargs = {"introduction": introduction} if introduction else {}
    register_oe_mode(
        f"{prefix}.direct",
        "direct",
        absolute_distance_stems,
        absolute_direct_answers,
        question_instruction=absolute_direct_instructions,
        **kwargs,
    )
    register_oe_mode(
        f"{prefix}.sentence",
        "sentence",
        absolute_distance_stems,
        absolute_sentence_answers,
        question_instruction=absolute_sentence_instructions,
        **kwargs,
    )
    register_oe_mode(
        f"{prefix}.free",
        "free",
        absolute_distance_stems,
        absolute_sentence_answers,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
        **kwargs,
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

    for polarity, stems, direct_ans, sentence_ans in (
        (
            "farthest",
            multiview_distance_farthest_questions,
            multiview_distance_farthest_direct_answers,
            multiview_distance_farthest_sentence_answers,
        ),
        (
            "closest",
            multiview_distance_closest_questions,
            multiview_distance_closest_direct_answers,
            multiview_distance_closest_sentence_answers,
        ),
    ):
        register_oe_mode(
            f"multiview_distance.{polarity}.direct",
            "direct",
            stems,
            direct_ans,
            introduction=multiview_distance_introduction,
            question_instruction=relative_oe_direct_instructions,
        )
        register_oe_mode(
            f"multiview_distance.{polarity}.reasoning",
            "reasoning",
            stems,
            sentence_ans,
            introduction=multiview_distance_introduction,
            question_instruction=relative_oe_reasoning_instructions,
        )
        register_oe_mode(
            f"multiview_distance.{polarity}.free",
            "free",
            stems,
            sentence_ans,
            introduction=multiview_distance_introduction,
            question_instruction=EMPTY_QUESTION_INSTRUCTION,
        )
    register_mcq(
        "multiview_distance.obj_cam_mcq",
        multiview_distance_obj_cam_mcq_questions,
        answers=multiview_distance_obj_cam_mcq_answers,
        letter_only=False,
        introduction=multiview_distance_introduction,
    )


register_structured_distance_templates()
