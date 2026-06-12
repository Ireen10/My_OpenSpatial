"""TEMP — delete this entire file after the monitor/TV label experiment."""

_MONITOR_PROMPT_ALIASES = {
    "monitor": "television",
}


def sam3_prompt_text_for_tag(tag: str) -> str:
    """Map dataset tag -> SAM3 text prompt. Only used at inference time."""
    key = str(tag).strip().lower()
    return _MONITOR_PROMPT_ALIASES.get(key, str(tag).strip())
