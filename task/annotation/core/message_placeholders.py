"""Ensure messages human lines have <image> count matching each turn (annotation post-process)."""

from __future__ import annotations

import re
from typing import Any, Dict, List

IMAGE_TAG_RE = re.compile(r"<image>\s*", re.I)


def _human_gpt_pair_indices(conv: List[dict]) -> List[tuple[int, int]]:
    """Return (human_idx, gpt_idx) for each QA turn in a flat conversation."""
    pairs: List[tuple[int, int]] = []
    i = 0
    while i < len(conv):
        if conv[i].get("from") == "human":
            gpt_idx = i + 1 if i + 1 < len(conv) and conv[i + 1].get("from") == "gpt" else i
            pairs.append((i, gpt_idx))
            i += 2
            continue
        i += 1
    return pairs


def _fix_human_at(conv: List[dict], human_idx: int, turn: dict) -> None:
    msg = conv[human_idx]
    n = max(1, int(turn.get("image_placeholder_count") or 1))
    prefix = (turn.get("question_prefix") or "").strip()
    body = IMAGE_TAG_RE.sub("", str(msg.get("value", ""))).strip()
    if prefix:
        if body.startswith(prefix):
            body = body[len(prefix):].lstrip()
        elif not body:
            body = ""
    img_prefix = " ".join(["<image>"] * n) + " "
    if prefix:
        msg["value"] = img_prefix + prefix + ("\n\n" + body if body else "")
    else:
        msg["value"] = img_prefix + body


def _sync_gpt_at(conv: List[dict], gpt_idx: int, turn: dict) -> None:
    answer = turn.get("answer_text")
    if answer is None or answer == "":
        return
    if gpt_idx < len(conv) and conv[gpt_idx].get("from") == "gpt":
        conv[gpt_idx]["value"] = str(answer).strip()


def _apply_turns_to_conversation(conv: List[dict], turn_records: List[dict]) -> None:
    pairs = _human_gpt_pair_indices(conv)
    for ti, tr in enumerate(turn_records):
        if ti >= len(pairs):
            break
        hi, gi = pairs[ti]
        _fix_human_at(conv, hi, tr)
        _sync_gpt_at(conv, gi, tr)


def _fix_human_in_conversation(conv: List[dict], turn: dict) -> None:
    pairs = _human_gpt_pair_indices(conv)
    if pairs:
        _fix_human_at(conv, pairs[0][0], turn)

def _placeholder_count_for_qa_item(item: Any) -> int:
    if isinstance(item, list):
        return max(1, len(item))
    return 1


def sync_messages_with_qa_images(
    messages: Any,
    qa_images: List[Any],
) -> Any:
    """Align <image> tags with QA_images length when turn_records are unavailable."""
    if not messages or not qa_images:
        return messages
    convs = [messages] if isinstance(messages[0], dict) else list(messages)
    for i, conv in enumerate(convs):
        if i >= len(qa_images) or not isinstance(conv, list):
            break
        _fix_human_in_conversation(
            conv,
            {"image_placeholder_count": _placeholder_count_for_qa_item(qa_images[i])},
        )
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        return convs[0]
    return convs


def sync_messages_with_turns(
    messages: Any,
    turn_records: List[dict],
) -> Any:
    """
    Post-process after create_messages_from_prompts.

    Each conversation (one QA before flatten) gets <image> tags equal to
    that turn's image_placeholder_count. Single-turn rows only need this at
    annotation time; aggregate merge just concatenates already-valid turns.
    """
    if not messages or not turn_records:
        return messages

    if isinstance(messages[0], dict):
        convs = [messages]
    else:
        convs = list(messages)

    for i, conv in enumerate(convs):
        if not isinstance(conv, list):
            continue
        if len(convs) == 1 and len(turn_records) > 1:
            _apply_turns_to_conversation(conv, turn_records)
            break
        if i >= len(turn_records):
            break
        _apply_turns_to_conversation(conv, [turn_records[i]])

    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        return convs[0]
    return convs
