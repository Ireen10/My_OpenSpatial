# Template IDs:
#   distance.absolute_*           — singleview only (no introduction)
#   multiview_distance.absolute_* — same stems as distance.absolute_* + introduction
#   multiview_distance.*          — multiview-only (farthest/closest, obj_cam)
#
# ─── Shared: absolute distance (OE) ─────────────────────────────────────

distance_template_questions_v2 = [
    "Measuring from the closest point of each object, what is the distance between the [A] and the [B] (in meters)?",
    "Measuring from the closest point of each object, what is the distance between the [A] and the [B] (in centimeters)?",
    "What is the distance between the [A] and the [B] (in meters)?",
    "What is the distance between the [A] and the [B] (in centimeters)?",
    "Consider the real-world 3D location of the objects. What is the distance between the [A] and the [B] (in meters)?",
    "Consider the real-world 3D location of the objects. What is the distance between the [A] and the [B] (in centimeters)?",
]

distance_template_questions_m = [q for q in distance_template_questions_v2 if "meters)" in q]
distance_template_questions_cm = [q for q in distance_template_questions_v2 if "centimeters)" in q]

absolute_m_direct_instructions = [
    "Give the distance in meters only.",
    "Reply with the numeric distance in meters.",
    "State their separation in meters as a single value.",
]

absolute_m_sentence_instructions = [
    "Please answer in meters.",
    "Provide your final answer in meters.",
    "Use meters as the unit in your answer.",
]

absolute_cm_direct_instructions = [
    "Give the distance in centimeters only.",
    "Reply with the numeric distance in centimeters.",
    "State their separation in centimeters as a single value.",
]

absolute_cm_sentence_instructions = [
    "Please answer in centimeters.",
    "Provide your final answer in centimeters.",
    "Use centimeters as the unit in your answer.",
]

absolute_m_direct_answers = ["[X] meters"]
absolute_m_sentence_answers = [
    "The distance between the [A] and the [B] is [X] meters.",
    "The [A] and the [B] are approximately [X] meters apart.",
]

absolute_cm_direct_answers = ["[X] centimeters"]
absolute_cm_sentence_answers = [
    "The distance between the [A] and the [B] is [X] centimeters.",
    "The [A] and the [B] are approximately [X] centimeters apart.",
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
    "[X].",
    "The [X] is farther from the [C].",
]

positional_far_oe_reasoning_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. Therefore, [X] is farther from [C].",
    "[X] is farther from [C]: the distance to [A] is about [D] and to [B] is about [E].",
]
positional_far_oe_reasoning_semantic_answers = [
    "The [X] is farther from the [C].",
    "[X] is farther from the [C].",
]

positional_far_mcq_direct_answers = ["[X]", "[X]."]
positional_far_mcq_reasoning_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. [X]",
]
positional_far_mcq_reasoning_semantic_answers = [
    "The [F] is farther from the [C]. [X]",
    "[F] is farther from the [C], so the answer is [X].",
]
positional_far_mcq_free_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. Therefore, the answer is [X].",
]
positional_far_mcq_free_semantic_answers = [
    "The [F] is farther from the [C]. Therefore, the answer is [X].",
    "[F] is farther from the [C], so the answer is [X].",
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
    "[X].",
    "The [X] is closer to the [C].",
]

positional_close_oe_reasoning_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. Therefore, [X] is closer to [C].",
    "[X] is closer to [C]: the distance to [A] is about [D] and to [B] is about [E].",
]
positional_close_oe_reasoning_semantic_answers = [
    "The [X] is closer to the [C].",
    "[X] is closer to the [C].",
]

positional_close_mcq_direct_answers = ["[X]", "[X]."]
positional_close_mcq_reasoning_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. [X]",
]
positional_close_mcq_reasoning_semantic_answers = [
    "The [F] is closer to the [C]. [X]",
    "[F] is closer to the [C], so the answer is [X].",
]
positional_close_mcq_free_metric_answers = [
    "The distance from [A] to [C] is about [D]. The distance from [B] to [C] is about [E]. Therefore, the answer is [X].",
]
positional_close_mcq_free_semantic_answers = [
    "The [F] is closer to the [C]. Therefore, the answer is [X].",
    "[F] is closer to the [C], so the answer is [X].",
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
    "[X] is the farthest from [Y].",
    "Among the listed objects, [X] has the greatest distance to [Y].",
    "The object [X] is most distant from [Y] across the views.",
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
    "[X] is the closest to [Y].",
    "Among the listed objects, [X] has the smallest distance to [Y].",
    "The object [X] is nearest to [Y] across the views.",
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
    "[X]",
]
from .register_structured import (
    EMPTY_QUESTION_INSTRUCTION,
    mixed_metric_default_profile,
    register_mcq,
    register_oe,
)


def _relative_oe_reasoning_profiles(polarity: str):
    if polarity == "far":
        return mixed_metric_default_profile(
            positional_far_oe_reasoning_metric_answers,
            positional_far_oe_reasoning_semantic_answers,
        )
    return mixed_metric_default_profile(
        positional_close_oe_reasoning_metric_answers,
        positional_close_oe_reasoning_semantic_answers,
    )


def _relative_mcq_reasoning_profiles(polarity: str):
    if polarity == "far":
        return mixed_metric_default_profile(
            positional_far_mcq_reasoning_metric_answers,
            positional_far_mcq_reasoning_semantic_answers,
        )
    return mixed_metric_default_profile(
        positional_close_mcq_reasoning_metric_answers,
        positional_close_mcq_reasoning_semantic_answers,
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
    """Register far/close OE templates: direct / reasoning / free (free = empty question_instruction)."""
    if polarity == "far":
        stems = positional_far_oe_questions
        direct_ans = positional_far_oe_direct_answers
    else:
        stems = positional_close_oe_questions
        direct_ans = positional_close_oe_direct_answers

    reasoning_profiles = _relative_oe_reasoning_profiles(polarity)

    register_oe(
        f"distance.relative_{polarity}.direct",
        stems,
        direct_ans,
        question_instruction=relative_oe_direct_instructions,
    )
    register_oe(
        f"distance.relative_{polarity}.reasoning",
        stems,
        answer_profiles=reasoning_profiles,
        question_instruction=relative_oe_reasoning_instructions,
    )
    register_oe(
        f"distance.relative_{polarity}.free",
        stems,
        answer_profiles=reasoning_profiles,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def _register_relative_mcq_pair(polarity: str) -> None:
    if polarity == "far":
        stems = positional_far_mcq_questions
        direct_ans = positional_far_mcq_direct_answers
    else:
        stems = positional_close_mcq_questions
        direct_ans = positional_close_mcq_direct_answers

    reasoning_profiles = _relative_mcq_reasoning_profiles(polarity)
    free_profiles = _relative_mcq_free_profiles(polarity)

    register_mcq(
        f"distance.relative_{polarity}_mcq.direct",
        stems,
        answers=direct_ans,
        letter_only=False,
        question_instruction=relative_mcq_direct_instructions,
    )
    register_mcq(
        f"distance.relative_{polarity}_mcq.reasoning",
        stems,
        answer_profiles=reasoning_profiles,
        letter_only=False,
        question_instruction=relative_mcq_reasoning_instructions,
    )
    register_mcq(
        f"distance.relative_{polarity}_mcq.free",
        stems,
        answer_profiles=free_profiles,
        letter_only=False,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def register_structured_distance_templates() -> None:
    # Absolute distance: singleview IDs (no introduction) vs multiview IDs (+ introduction)
    for unit, stems, direct_ans, sentence_ans, direct_instr, sentence_instr in (
        (
            "m",
            distance_template_questions_m,
            absolute_m_direct_answers,
            absolute_m_sentence_answers,
            absolute_m_direct_instructions,
            absolute_m_sentence_instructions,
        ),
        (
            "cm",
            distance_template_questions_cm,
            absolute_cm_direct_answers,
            absolute_cm_sentence_answers,
            absolute_cm_direct_instructions,
            absolute_cm_sentence_instructions,
        ),
    ):
        register_oe(
            f"distance.absolute_{unit}.direct",
            stems,
            direct_ans,
            question_instruction=direct_instr,
        )
        register_oe(
            f"distance.absolute_{unit}.sentence",
            stems,
            sentence_ans,
            question_instruction=sentence_instr,
        )
        register_oe(
            f"multiview_distance.absolute_{unit}.direct",
            stems,
            direct_ans,
            introduction=multiview_distance_introduction,
            question_instruction=direct_instr,
        )
        register_oe(
            f"multiview_distance.absolute_{unit}.sentence",
            stems,
            sentence_ans,
            introduction=multiview_distance_introduction,
            question_instruction=sentence_instr,
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
        register_oe(
            f"multiview_distance.{polarity}.direct",
            stems,
            direct_ans,
            introduction=multiview_distance_introduction,
            question_instruction=relative_oe_direct_instructions,
        )
        register_oe(
            f"multiview_distance.{polarity}.reasoning",
            stems,
            sentence_ans,
            introduction=multiview_distance_introduction,
            question_instruction=relative_oe_reasoning_instructions,
        )
        register_oe(
            f"multiview_distance.{polarity}.free",
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
