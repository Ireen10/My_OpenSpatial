# Template IDs: multiview_correspondence.point2point[_{num}].oe.{sentence|free};
#   mcq.{direct|sentence|free}

correspondence_introduction = [
    "The two images depict the same scene captured from different camera viewpoints.",
    "Both images show the same environment photographed from different viewing angles.",
    "You are given two views of one scene, taken from different perspectives.",
    "These two pictures represent the same space, each from a distinct viewpoint.",
    "The pair of images comes from the same scene under different camera poses.",
]

# OE: candidates are visible labels on image 2 only (no inline Options block).
point2point_oe_stems = [
    "The first image shows a point marked in [A] color. The second image shows several [B] labeled points. Which one corresponds to the original?",
    "In image one, a point is highlighted in [A] color. In the second image, multiple [B] labeled points are shown. Identify the corresponding point.",
    "The first image marks a point in [A] color. The second image presents several [B] labeled points. Which one matches the original?",
    "The first image shows a point highlighted in [A] color. The second image reveals several [B] labeled points. Which point matches the original?",
    "The first image features a point indicated in [A] color. Multiple [B] labeled points appear in the second image. Which one matches the original?",
    "In image one, a point is indicated in [A] color. In the second image, there are several [B] labeled points. Can you identify the corresponding point?",
]

point2point_oe_stems_num = [
    "The first image shows a point marked in [A] color. The second image shows several [B] points labeled 1–4. Which one corresponds to the original?",
    "In image one, a point is highlighted in [A] color. In the second image, multiple [B] points labeled 1, 2, 3, 4 are shown. Identify the corresponding point.",
    "The first image marks a point in [A] color. The second image presents several [B] points labeled 1–4. Which one matches the original?",
    "The first image shows a point highlighted in [A] color. The second image reveals several [B] points labeled 1, 2, 3, 4. Which point matches the original?",
    "The first image features a point indicated in [A] color. Multiple [B] points labeled 1–4 appear in the second image. Which one matches the original?",
    "In image one, a point is indicated in [A] color. In the second image, there are several [B] points labeled 1, 2, 3, 4. Can you identify the corresponding point?",
]

# MCQ: options inserted via [O] (handler fills shared["O"]).
point2point_mcq_stems = [
    "The first image shows a point marked in [A] color. Which [B] point in the second image is the corresponding match? [O]",
    "In image one, a point is highlighted in [A] color. Select the matching point from the second image. [O]",
    "The first image marks a point in [A] color. Which option identifies the corresponding [B] point in image two? [O]",
    "A point is highlighted in [A] on the first image. Choose the corresponding point on the second image. [O]",
    "Image one shows a [A] marked query point. Which labeled point on image two corresponds? [O]",
]

point2point_mcq_stems_num = [
    "The first image shows a point marked in [A] color. Which option identifies the corresponding [B] point on image two? [O]",
    "In image one, a point is highlighted in [A] color. Select the matching point from the options. [O]",
    "The first image marks a point in [A] color. Which [B] point in the second image is the match? [O]",
    "A point is indicated in [A] on the first image. Choose the corresponding point from the list. [O]",
    "Image one shows a [A] query point. Pick the matching point on image two. [O]",
]

correspondence_oe_sentence_instructions = [
    "Answer in a complete sentence.",
    "Reply using a full sentence.",
    "Answer in a complete sentence that names the matching point.",
    "Reply with a full sentence identifying the corresponding point.",
    "Give your answer as a complete sentence naming the matching point.",
]

correspondence_mcq_sentence_instructions = [
    "Answer in a complete sentence.",
    "Reply using a full sentence.",
    "Answer in a complete sentence that states the correct option.",
    "Reply with a full sentence identifying the matching option.",
    "Give your answer as a complete sentence naming the chosen option.",
]

# [A]=query color, [B]=candidate color; [L]=point label only (point-A / point-1);
# [E]=MCQ option letter (always A–D, matches Options line); [T]=full option text (A. point-A / A. point-1).
correspondence_oe_sentence_answers = [
    "The point [L] in image 2 corresponds to the [A] marked point in image 1.",
    "In image 2, [L] matches the [A] query point shown in image 1.",
    "The [B] point [L] in the second image is the same location as the [A] point in the first image.",
    "Image 2's [L] aligns with the [A] highlighted point in image 1.",
]

correspondence_mcq_sentence_answers = [
    "In image 2, [L] matches the [A] query point shown in image 1. Therefore the correct option is [E].",
    "The point [L] in image 2 corresponds to the [A] marked point in image 1. So the answer is [E].",
    "The [B] point [L] in the second image is the same location as the [A] point in image 1. The correct option is [E].",
    "Image 2's [L] aligns with the [A] highlighted point in image 1. Therefore the answer is [E].",
]

from ..annotation.core.structured_prompt_template import AnswerInstructionProfile
from .register_structured import (
    EMPTY_QUESTION_INSTRUCTION,
    MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS,
    register_mcq_mode,
    register_oe,
)

_CORRESPONDENCE_OE_SENTENCE_PROFILE = AnswerInstructionProfile(
    "sentence",
    answer_templates=correspondence_oe_sentence_answers,
)
_CORRESPONDENCE_OE_FREE_PROFILE = AnswerInstructionProfile(
    "free",
    answer_templates=correspondence_oe_sentence_answers,
)


def _register_point2point_family(
    suffix: str,
    oe_stems: list,
    mcq_stems: list,
) -> None:
    base = f"multiview_correspondence.point2point{suffix}"
    for mode, qinstr, profile in (
        ("sentence", correspondence_oe_sentence_instructions, _CORRESPONDENCE_OE_SENTENCE_PROFILE),
        ("free", EMPTY_QUESTION_INSTRUCTION, _CORRESPONDENCE_OE_FREE_PROFILE),
    ):
        register_oe(
            f"{base}.oe.{mode}",
            oe_stems,
            [],
            introduction=correspondence_introduction,
            question_instruction=qinstr,
            answer_profiles={mode: profile},
            instruction_types=[mode],
        )
    for mode, qinstr, answers in (
        ("direct", MCQ_ANSWER_WITH_OPTION_TEXT_INSTRUCTIONS, ["[T]"]),
        ("sentence", correspondence_mcq_sentence_instructions, correspondence_mcq_sentence_answers),
        ("free", EMPTY_QUESTION_INSTRUCTION, correspondence_mcq_sentence_answers),
    ):
        register_mcq_mode(
            f"{base}.mcq.{mode}",
            mode,
            mcq_stems,
            answers=answers,
            introduction=correspondence_introduction,
            question_instruction=qinstr,
        )


def register_structured_correspondence_templates() -> None:
    _register_point2point_family("", point2point_oe_stems, point2point_mcq_stems)
    _register_point2point_family("_num", point2point_oe_stems_num, point2point_mcq_stems_num)


register_structured_correspondence_templates()
