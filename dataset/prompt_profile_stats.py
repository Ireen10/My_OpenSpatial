"""Turn-level prompt profile keys for dataset export statistics."""

from __future__ import annotations

from typing import Any, Dict, Optional

PROFILE_MODE_SUFFIXES = frozenset(
    {
        "direct",
        "reasoning",
        "free",
        "sentence",
        "true",
        "false",
        "letter_only",
    }
)

_LEGACY_MODE_ALIASES = {
    "default": "direct",
    "letter_only": "direct",
}
# Raw profile keys that alias to direct; template_id suffix wins when present.
_LEGACY_ALIAS_KEYS = frozenset(_LEGACY_MODE_ALIASES.keys())


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


def _normalize_mode(mode: str) -> str:
    m = (mode or "").strip()
    if not m:
        return "none"
    return _LEGACY_MODE_ALIASES.get(m, m)


def _mode_from_template_id(template_id: str) -> Optional[str]:
    tid = (template_id or "").strip()
    if not tid:
        return None
    last = tid.rsplit(".", 1)[-1]
    if last in PROFILE_MODE_SUFFIXES:
        return _normalize_mode(last)
    return None


def prompt_profile_stat_key(turn: Dict[str, Any]) -> str:
    """
    Histogram key: ``{question_type}_{mode}`` (e.g. ``MCQ_direct``).

    Mode priority: ``constraint_mode`` > last ``template_id`` segment >
    ``instruction_type`` (never use judgment answer polarity true/false).
    """
    ps = turn.get("prompt_struct")
    if not isinstance(ps, dict):
        ps = {}

    qt = normalize_question_type_enum(
        str(ps.get("question_type_enum") or turn.get("question_type") or "unknown")
    )

    mode: Optional[str] = None
    template_id = str(ps.get("template_id") or "")
    mode_from_tid = _mode_from_template_id(template_id)

    cm = ps.get("constraint_mode")
    if isinstance(cm, str) and cm.strip():
        raw_cm = cm.strip()
        # Legacy answer profiles (default / letter_only) do not encode the sampled
        # constraint mode; prefer the template_id suffix when available.
        if raw_cm in _LEGACY_ALIAS_KEYS and mode_from_tid:
            mode = mode_from_tid
        else:
            mode = _normalize_mode(raw_cm)

    if mode is None:
        mode = mode_from_tid

    if mode is None:
        it = (ps.get("instruction_type") or "").strip()
        if qt == "judgment" and it in ("true", "false"):
            mode = "free"
        elif it:
            mode = _normalize_mode(it)
        else:
            mode = "none"

    if qt == "judgment" and mode in ("true", "false"):
        mode = "free"

    return f"{qt}_{mode}"
