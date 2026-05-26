"""Turn-level prompt profile keys for dataset export statistics."""

from __future__ import annotations

from typing import Any, Dict, Optional

PROFILE_MODE_SUFFIXES = frozenset(
    {
        "direct",
        "reasoning",
        "free",
        "sentence",
        "default",
        "true",
        "false",
        "letter_only",
    }
)


def normalize_question_type_enum(raw: str) -> str:
    q = (raw or "").strip()
    if not q:
        return "unknown"
    upper = q.upper()
    if upper == "MCQ":
        return "MCQ"
    if upper in ("OE", "OPEN_ENDED"):
        return "OE"
    if upper in ("JUDGMENT", "TRUE_FALSE"):
        return "judgment"
    return q


def _mode_from_template_id(template_id: str) -> Optional[str]:
    tid = (template_id or "").strip()
    if not tid:
        return None
    last = tid.rsplit(".", 1)[-1]
    if last in PROFILE_MODE_SUFFIXES:
        return last
    return None


def prompt_profile_stat_key(turn: Dict[str, Any]) -> str:
    """
    Histogram key: ``{question_type}_{mode}`` (e.g. ``MCQ_direct``).

    Mode prefers the last ``template_id`` segment (``distance.relative_far.direct``);
    otherwise ``prompt_struct.instruction_type`` (``letter_only`` → ``direct``).
    """
    ps = turn.get("prompt_struct")
    if not isinstance(ps, dict):
        ps = {}

    qt = normalize_question_type_enum(
        str(ps.get("question_type_enum") or turn.get("question_type") or "unknown")
    )
    mode = _mode_from_template_id(str(ps.get("template_id") or ""))
    if mode is None:
        it = (ps.get("instruction_type") or "").strip()
        if it == "letter_only":
            mode = "direct"
        elif it:
            mode = it
        else:
            mode = "none"
    if mode == "letter_only":
        mode = "direct"
    return f"{qt}_{mode}"
