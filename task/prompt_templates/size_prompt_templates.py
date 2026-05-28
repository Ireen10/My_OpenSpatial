# size.* / size.{big,small}.single_view — singleview (no multiview introduction)
# multiview_size.* — multiview pair judgment + N-ary superlative (all with introduction)

#### Size Predicate Templates for single-view images ####

size_predicate_questions_single_view = [
    "What is the length of the dimension that is largest in size (length, width, or height) of the [A]?",
    "What is the measurement for the longest side (length, width, or height) of the [A]?",
    "Can you provide the size of the [A]'s largest dimension (length, width, or height)?",
    "What is the length of the dimension that is maximum (length, width, or height) of the [A]?",
    "What is the length of the dimension that is the greatest (length, width, or height) of the [A]?",
    "What is the measurement of the [A]'s longest dimension (length, width, or height)?",
    "Can you tell me the size of the [A]'s maximum dimension (length, width, or height)?",
    "What is the length of the dimension that is the most extensive (length, width, or height) of the [A]?",
    "What is the measurement of the [A]'s greatest dimension (length, width, or height)?",
    "Can you provide the size of the [A]'s most significant dimension (length, width, or height)?",
]

size_answers_single_view = [
    "[X].",
    "The largest dimension of the [A] is [X].",
    "The [A] measures approximately [X] at its longest side.",
]

height_predicate_questions_single_view = [
    "Could you estimate the height of the [A]?",
    "What is the vertical measurement of the [A]?",
    "Can you provide the height dimension of the [A]?",
    "How tall does the [A] stand?",
    "What is the height of the [A]?",
    "Could you tell me the vertical size of the [A]?",
    "What is the measurement of the [A]'s height?",
    "Can you estimate how high the [A] is?",
    "What is the vertical dimension of the [A]?",
]

height_answers_single_view = [
    "[X].",
    "The height of the [A] is [X].",
    "The [A] stands approximately [X] tall.",
]
#### Single-view relative size predicate templates ####

big_predicate_questions_single_view = [
    "Is the [A] bigger than the [B]?",
    "Does the [A] have a larger size compared to the [B]?",
    "Can you confirm if the [A] is bigger than the [B]?",
]

big_true_responses_single_view = [
    "Yes, the [A] is bigger than the [B].",
    "Indeed, the [A] has a larger size compared to the [B].",
    "Correct, the [A] is larger in size than the [B].",
]

big_false_responses_single_view = [
    "No, the [A] is not bigger than the [B].",
    "Actually, the [A] might be smaller or the same size as the [B].",
    "Incorrect, the [A] is not larger than the [B].",
]

small_predicate_questions_single_view = [
    "Is the [A] smaller than the [B]?",
    "Does the [A] have a smaller size compared to the [B]?",
    "Can you confirm if the [A] is smaller than the [B]?",
]

small_true_responses_single_view = [
    "Yes, the [A] is smaller than the [B].",
    "Indeed, the [A] has a smaller size compared to the [B].",
    "Correct, the [A] occupies less space than the [B].",
]

small_false_responses_single_view = [
    "No, the [A] is not smaller than the [B].",
    "Actually, the [A] might be larger or the same size as the [B].",
    "Incorrect, the [A] is not smaller in size than the [B].",
]


#### Multiview size (pair judgment + N-ary superlative) ####

big_predicate_questions_multi_view = [
    "Is the [A] bigger than the [B]?",
    "Does the [A] have a larger size compared to the [B]?",
    "Can you confirm if the [A] is bigger than the [B]?",
]

big_true_responses_multi_view = [
    "Yes",
    "Correct",
    "Yes, the [A] is bigger than the [B].",
    "Indeed, the [A] has a larger size compared to the [B].",
    "Correct, the [A] is larger in size than the [B].",
]

big_false_responses_multi_view = [
    "No",
    "Incorrect",
    "No, the [A] is not bigger than the [B].",
    "Actually, the [A] might be smaller or the same size as the [B].",
    "Incorrect, the [A] is not larger than the [B].",
]

small_predicate_questions_multi_view = [
    "Is the [A] smaller than the [B]?",
    "Does the [A] have a smaller size compared to the [B]?",
    "Can you confirm if the [A] is smaller than the [B]?",
]

small_true_responses_multi_view = [
    "Yes",
    "Correct",
    "Yes, the [A] is smaller than the [B].",
    "Indeed, the [A] has a smaller size compared to the [B].",
    "Correct, the [A] occupies less space than the [B].",
]

small_false_responses_multi_view = [
    "No",
    "Incorrect",
    "No, the [A] is not smaller than the [B].",
    "Actually, the [A] might be larger or the same size as the [B].",
    "Incorrect, the [A] is not smaller in size than the [B].",
]



multiview_size_biggest_questions = [
    "Among the objects [T], which one is the biggest?",
    "Considering the set of objects [T], which has the largest size?",
    "From the objects [T], which one has the greatest size?",
    "Out of the objects [T], which one is the largest in size?",
    "Which object in [T] is the biggest?",
]

multiview_size_biggest_sentence_answers = [
    "The [X] is the biggest among the objects.",
    "Among the objects, the [X] is larger than any of the others.",
    "In terms of size, the [X] is the biggest one.",
]

multiview_size_smallest_questions = [
    "Among the objects [T], which one is the smallest?",
    "Considering the set of objects [T], which has the smallest size?",
    "From the objects [T], which one has the least size?",
    "Out of the objects [T], which one is the smallest in size?",
    "Which object in [T] is the smallest?",
]

multiview_size_smallest_sentence_answers = [
    "The [X] is the smallest among the objects.",
    "Among the objects, the [X] is smaller than any of the others.",
    "In terms of size, the [X] is the smallest one.",
]
from .register_structured import (
    EMPTY_QUESTION_INSTRUCTION,
    MULTIVIEW_SCENE_INTRODUCTION,
    register_oe_mode_family,
    register_judgment,
)

_UNCONSTRAINED = EMPTY_QUESTION_INSTRUCTION
multiview_size_introduction = MULTIVIEW_SCENE_INTRODUCTION

size_absolute_direct_instructions = [
    "Give the numeric measurement with the appropriate unit (meters or centimeters).",
    "Reply with the measurement using meters or centimeters as appropriate.",
    "State the value as a single number with unit.",
]

size_absolute_sentence_instructions = [
    "Answer in a complete sentence that includes the measurement and unit.",
    "Reply using a full sentence with the measurement and appropriate unit.",
    "Provide your final answer in a complete sentence including the measurement.",
]

size_absolute_direct_answers = ["[X]"]


def _register_size_absolute_family(kind: str, stems: list, sentence_answers: list) -> None:
    prefix = f"size.{kind}.single_view"
    register_oe_mode_family(
        prefix,
        stems,
        sentence_answers,
        direct_answers=size_absolute_direct_answers,
        direct_instructions=size_absolute_direct_instructions,
        sentence_instructions=size_absolute_sentence_instructions,
    )


def register_structured_size_templates() -> None:
    _register_size_absolute_family(
        "absolute", size_predicate_questions_single_view, size_answers_single_view,
    )
    _register_size_absolute_family(
        "height", height_predicate_questions_single_view, height_answers_single_view,
    )

    register_judgment(
        "size.big.single_view",
        big_predicate_questions_single_view,
        big_true_responses_single_view,
        big_false_responses_single_view,
        question_instruction=_UNCONSTRAINED,
    )
    register_judgment(
        "size.small.single_view",
        small_predicate_questions_single_view,
        small_true_responses_single_view,
        small_false_responses_single_view,
        question_instruction=_UNCONSTRAINED,
    )
    register_judgment(
        "multiview_size.big.pair",
        big_predicate_questions_multi_view,
        big_true_responses_multi_view,
        big_false_responses_multi_view,
        introduction=multiview_size_introduction,
        question_instruction=_UNCONSTRAINED,
    )
    register_judgment(
        "multiview_size.small.pair",
        small_predicate_questions_multi_view,
        small_true_responses_multi_view,
        small_false_responses_multi_view,
        introduction=multiview_size_introduction,
        question_instruction=_UNCONSTRAINED,
    )
    for polarity, stems, sentence_ans in (
        (
            "biggest",
            multiview_size_biggest_questions,
            multiview_size_biggest_sentence_answers,
        ),
        (
            "smallest",
            multiview_size_smallest_questions,
            multiview_size_smallest_sentence_answers,
        ),
    ):
        register_oe_mode_family(
            f"multiview_size.{polarity}",
            stems,
            sentence_ans,
            direct_answers=["[X]."],
            direct_instructions=_UNCONSTRAINED,
            sentence_instructions=_UNCONSTRAINED,
            introduction=multiview_size_introduction,
        )


register_structured_size_templates()
