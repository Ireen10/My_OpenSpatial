# depth.<family>.{direct|sentence|free} — question-side instruction constraints only.

depth_ordering_questions = [
    "Given the [T] [A], please order them by depth (from near to far).",
    "Please arrange the [T] [A] based on their depth (from near to far).",
    "Order the [T] [A] according to their depth from near to far.",
    "Sort the [T] [A] by depth (from near to far).",
    "Can you organize the [T] [A] in order of their depth (from near to far)?",
    "Please sequence the [T] [A] from shallowest to deepest .",
]

depth_ordering_answers = [
    "[X].",
    "The order is [X].",
    "From near to far: the order can be represented as [X].",
    "In order from near to far: this can be expressed as [X].",
    "Arranged from near to far: the configuration can be denoted as [X].",
    "From closest to farthest: the arrangement can be represented as [X].",
    "Ordered from near to far: this can be illustrated as [X].",
]

depth_ordering_questions_mcq = [
    "Given the [T] [X], please order them by depth (from near to far). Consider the following options: [Y] and choose the correct one.",
    "Please arrange the [T] [X] based on their depth (from near to far). Please consider the following options: [Y], and choose the correct one. ",
    "Order the [T] [X] according to their depth from near to far. Think about these options: [Y]. Which one do you believe is correct?",
    "Sort the [T] [X] by depth (from near to far). Here are the options to choose from: [Y]. Please select the correct answer.",
    "Can you organize the [T] [X] in order of their depth (from near to far)? Consider these options: [Y], and choose the correct answer.",
    "Please sequence the [T] [X] from shallowest to deepest . Before making a decision, please review the following options: [Y], and select the correct one."
]

depth_ordering_answers_mcq = [
    "[X].",
    "The answer is [X].",
    "In order from near to far: the answer is given in option [X].",
    "Arranged from near to far: the answer is denoted as option [X].",
    "From closest to farthest: the answer is represented as option [X].",
    "Ordered from near to far: the answer is option [X].",
]


depth_choice_questions = [
    "Between the [T] [A], which one is the [B] closest to the camera?",
    "Among the [T] [A], which one is the [B] nearest to the camera?",
    "From the [T] [A], identify the one that is the [B] closest to the camera.",
    "Considering the [T] [A], which one is the [B] nearest to the camera?",
    "Out of the [T] [A], which one has the [B] smallest depth?",
]

depth_choice_answers = [
    "[X].",
    "The [T] [X] is the one [B] closest to the camera.",
    "Among the objects, the [T] [X] is the [B] closest to the camera.",
]


depth_choice_questions_mcq = [
    "Between the [T] [X], which one is the [Y] closest to the camera? Consider the following options: [Z] and choose the correct one.",
    "Among the [T] [X], which one is the [Y] nearest to the camera? Please consider the following options: [Z], and choose the correct one. ",
    "From the [T] [X], identify the one that is the [Y] closest to the camera. Think about these options: [Z]. Which one do you believe is correct?",
    "Considering the [T] [X], which one is the [Y] nearest to the camera? Here are the options to choose from: [Z]. Please select the correct answer.",
    "Out of the [T] [X], which one has the [Y] smallest depth? Consider these options: [Z], and choose the correct answer.",
]

depth_choice_answers_mcq = [
    "[X].",
    "The answer is [X].",
]


depth_farthest_questions = [
    "Between the [T] [A], which one is the farthest from the camera?",
    "Among the [T] [A], which one is the most distant from the camera?",
    "From the [T] [A], identify the one that is the farthest from the camera.",
    "Considering the [T] [A], which one is the most distant from the camera?",
    "Out of the [T] [A], which one has the greatest depth?",
    "From the [T] [A], which is the one with the largest depth?",
]
depth_farthest_answers = [
    "[X].",
    "The [T] [X] is the farthest from the camera.",
    "Among the objects, the [T] [X] is farther from the camera than any of them.",
    "The [T] [X] has the greatest depth.",
    "The [T] [X] is the one most distant from the camera.",
]

depth_farthest_questions_mcq = [
    "Between the [T] [X], which one is the farthest from the camera? Consider the following options: [Y] and choose the correct one.",
    "Among the [T] [X], which one is the most distant from the camera? Please consider the following options: [Y], and choose the correct one.",
    "From the [T] [X], identify the one that is the farthest from the camera. Think about these options: [Y]. Which one do you believe is correct?",
    "Considering the [T] [X], which one is the most distant from the camera? Here are the options to choose from: [Y]. Please select the correct answer.",
    "Out of the [T] [X], which one has the greatest depth? Consider these options: [Y], and choose the correct answer.",
    "From the [T] [X], which one is the one with the largest depth? Before making a decision, please review the following options: [Y], and select the correct one."
]

depth_farthest_answers_mcq = [
    "[X].",
    "The answer is [X].",
]


depth_closest_questions = [
    "Between the [T] [A], which one is the closest to the camera?",
    "Among the [T] [A], which one is the nearest to the camera?",
    "From the [T] [A], identify the one that is the closest to the camera.",
    "Considering the [T] [A], which one is the nearest to the camera?",
    "Out of the [T] [A], which one has the smallest depth?",
    "From the [T] [A], which one is the one with the least depth?",
]

depth_closest_answers = [
    "[X].",
    "The [T] [X] is the closest to the camera.",
    "Among the objects, the [T] [X] is closer to the camera than any of them.",
    "The [T] [X] has the smallest depth.",
    "The [T] [X] is the one nearest to the camera.",
]

depth_closest_questions_mcq = [
    "Between the [T] [X], which one is the closest to the camera? Consider the following options: [Y] and choose the correct one.",
    "Among the [T] [X], which one is the nearest to the camera? Please consider the following options: [Y], and choose the correct one. ",
    "From the [T] [X], identify the one that is the closest to the camera. Think about these options: [Y]. Which one do you believe is correct?",
    "Considering the [T] [X], which one is the nearest to the camera? Here are the options to choose from: [Y]. Please select the correct answer.",
    "Out of the [T] [X], which one has the smallest depth? Consider these options: [Y], and choose the correct answer.",
    "From the [T] [X], which one is the one with the least depth? Before making a decision, please review the following options: [Y], and select the correct one."
]

depth_closest_answers_mcq = [
    "[X].",
    "The answer is [X].",
]
from .register_structured import (
    EMPTY_QUESTION_INSTRUCTION,
    register_mcq_mode,
    register_oe_mode,
)

depth_direct_instructions = [
    "Answer with the required label, ordering, or option only.",
    "Give a concise answer without extra explanation.",
    "Reply using only the label(s) or option identifier needed.",
]

depth_sentence_instructions = [
    "Answer in a complete sentence.",
    "Reply using a full sentence.",
    "Please describe your answer in a complete sentence.",
]


def _direct_answers(pool: list) -> list:
    return [pool[0]] if pool else ["[X]"]


def _register_depth_oe_family(
    base_id: str,
    stems: list,
    all_answers: list,
) -> None:
    sentence_answers = all_answers
    direct_answers = _direct_answers(all_answers)
    register_oe_mode(
        f"{base_id}.direct",
        "direct",
        stems,
        direct_answers,
        question_instruction=depth_direct_instructions,
    )
    register_oe_mode(
        f"{base_id}.sentence",
        "sentence",
        stems,
        sentence_answers,
        question_instruction=depth_sentence_instructions,
    )
    register_oe_mode(
        f"{base_id}.free",
        "free",
        stems,
        sentence_answers,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def _register_depth_mcq_family(
    base_id: str,
    stems: list,
    all_answers: list,
) -> None:
    sentence_answers = all_answers
    direct_answers = _direct_answers(all_answers)
    register_mcq_mode(
        f"{base_id}.direct",
        "direct",
        stems,
        answers=direct_answers,
        question_instruction=depth_direct_instructions,
    )
    register_mcq_mode(
        f"{base_id}.sentence",
        "sentence",
        stems,
        answers=sentence_answers,
        question_instruction=depth_sentence_instructions,
    )
    register_mcq_mode(
        f"{base_id}.free",
        "free",
        stems,
        answers=sentence_answers,
        question_instruction=EMPTY_QUESTION_INSTRUCTION,
    )


def register_structured_depth_templates() -> None:
    _register_depth_oe_family("depth.ordering", depth_ordering_questions, depth_ordering_answers)
    _register_depth_mcq_family(
        "depth.ordering_mcq", depth_ordering_questions_mcq, depth_ordering_answers_mcq,
    )
    _register_depth_oe_family("depth.choice", depth_choice_questions, depth_choice_answers)
    _register_depth_mcq_family(
        "depth.choice_mcq", depth_choice_questions_mcq, depth_choice_answers_mcq,
    )
    _register_depth_oe_family("depth.farthest", depth_farthest_questions, depth_farthest_answers)
    _register_depth_mcq_family(
        "depth.farthest_mcq", depth_farthest_questions_mcq, depth_farthest_answers_mcq,
    )
    _register_depth_oe_family("depth.closest", depth_closest_questions, depth_closest_answers)
    _register_depth_mcq_family(
        "depth.closest_mcq", depth_closest_questions_mcq, depth_closest_answers_mcq,
    )


register_structured_depth_templates()
