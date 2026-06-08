"""
Message creation utilities for annotation tasks.

Extracts the duplicated create_messages_from_prompts patterns.
"""


def create_singleview_messages(prompts):
    """
    Create messages for singleview tasks.

    Supports both single-turn and multi-turn prompts:
    - str prompt: single QA turn  → [human, gpt]
    - list[str] prompt: multi-turn → [human, gpt, human, gpt, ...] (only first turn gets <image>)

    Args:
        prompts: list of (str | list[str]), each str containing "Question ... Answer: Answer"

    Returns:
        list of message lists: [[{"from": "human", ...}, {"from": "gpt", ...}, ...], ...]
    """
    messages = []
    for prompt in prompts:
        if isinstance(prompt, list):
            msg = _build_multi_turn(prompt, num_images=1)
            if msg:
                messages.append(msg)
        else:
            msg = _split_single_prompt(prompt, num_images=1)
            if msg:
                messages.append(msg)
    return messages


def create_multiview_messages(prompts, processed_images):
    """
    Create messages for multiview tasks.

    Supports both single-turn and multi-turn prompts:
    - str prompt: single QA turn with N <image> tags
    - list[str] prompt: multi-turn with N <image> tags only on first turn

    Args:
        prompts: list of (str | list[str])
        processed_images: list of lists, each inner list = images for that prompt

    Returns:
        list of message lists
    """
    messages = []
    for i, prompt in enumerate(prompts):
        slot = processed_images[i] if i < len(processed_images) else None
        num_images = len(slot) if isinstance(slot, list) else 1
        if isinstance(prompt, list):
            msg = _build_multi_turn(prompt, num_images=num_images)
            if msg:
                messages.append(msg)
        else:
            msg = _split_single_prompt(prompt, num_images=num_images)
            if msg:
                messages.append(msg)
    return messages


def _build_multi_turn(sub_prompts, num_images=1):
    """
    Build a multi-turn message from a list of "question Answer: answer" strings.

    Only the first turn gets <image> tag(s).
    Returns a flat list: [human, gpt, human, gpt, ...] or None if empty.
    """
    message = []
    for i, sub_prompt in enumerate(sub_prompts):
        if "Answer: " not in sub_prompt:
            continue
        question, answer = sub_prompt.split("Answer: ", 1)
        question = question.strip()
        answer = answer.strip()
        if i == 0:
            prefix = " ".join(["<image>"] * num_images) + " "
            question = prefix + question
        message.append({"from": "human", "value": question})
        message.append({"from": "gpt", "value": answer})
    return message if message else None


def _split_single_prompt(prompt, num_images=1):
    """
    Helper: split a single "question Answer: answer" string into a message list.

    Returns None if no "Answer: " found.
    """
    if "Answer: " not in prompt:
        return None
    question, answer = prompt.split("Answer: ", 1)
    question = question.strip()
    answer = answer.strip()
    prefix = " ".join(["<image>"] * num_images) + " "
    question = prefix + question
    return [
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer},
    ]
