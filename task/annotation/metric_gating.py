"""Instruction-mode sampling and metric-depth guards for annotation prompts."""

from __future__ import annotations

from typing import Collection, List, Optional, Sequence

from task.annotation.core.thread_rng import rng

FREE_INSTRUCTION_MODE = "free"
DEFAULT_FREE_WEIGHT = 0.6

# Stems that explicitly ask for numeric distance estimation (relative distance pools).
RELATIVE_DISTANCE_STEM_METRIC_MARKERS = (
    "Estimate the real-world distances",
)

RELATIVE_DISTANCE_MODES = ("direct", "reasoning", "sentence", "free")
ABSOLUTE_DISTANCE_MODES = ("direct", "sentence", "free")

_METER_THRESHOLD = 1.0


def format_distance_value(dist_m: float, *, scaling_factor: float = 1.0) -> str:
    """Format distance for answer binding: cm if < 1 m after scaling, else meters."""
    meters = float(dist_m) * float(scaling_factor)
    if meters < _METER_THRESHOLD:
        return f"{meters * 100:.2f} centimeters"
    return f"{meters:.2f} meters"


def pick_relative_distance_mode(*, is_metric_depth: bool) -> str:
    """Relative OE/MCQ: four modes; reasoning only when metric depth is available."""
    return pick_instruction_mode(
        RELATIVE_DISTANCE_MODES,
        is_metric_depth=is_metric_depth,
        metric_only_modes=("reasoning",),
    )


def eligible_instruction_modes(
    modes: Sequence[str],
    *,
    is_metric_depth: Optional[bool] = None,
    metric_only_modes: Collection[str] = ("reasoning",),
) -> List[str]:
    """Modes allowed for this sample; drop metric-only modes when depth is not metric."""
    if is_metric_depth is False:
        return [m for m in modes if m not in metric_only_modes]
    return list(modes)


def pick_instruction_mode(
    modes: Sequence[str],
    *,
    is_metric_depth: Optional[bool] = None,
    metric_only_modes: Collection[str] = ("reasoning",),
    free_mode: str = FREE_INSTRUCTION_MODE,
    free_weight: float = DEFAULT_FREE_WEIGHT,
) -> str:
    """
    Sample an instruction constraint mode.

    When ``free_mode`` is among eligible modes: ``free_weight`` (default 60%) for free,
    remaining probability split equally across other eligible modes.
    Otherwise: uniform over eligible modes.
    """
    eligible = eligible_instruction_modes(
        modes,
        is_metric_depth=is_metric_depth,
        metric_only_modes=metric_only_modes,
    )
    if not eligible:
        raise ValueError(
            f"no eligible instruction mode (is_metric_depth={is_metric_depth!r}, "
            f"modes={list(modes)!r})"
        )
    if len(eligible) == 1:
        return eligible[0]

    if free_mode in eligible:
        others = [m for m in eligible if m != free_mode]
        if not others:
            return free_mode
        per_other = (1.0 - free_weight) / len(others)
        weights = [
            free_weight if m == free_mode else per_other for m in eligible
        ]
        return rng().choices(eligible, weights=weights, k=1)[0]

    return rng().choice(eligible)


def pick_metric_gated_mode(
    modes: Sequence[str],
    *,
    is_metric_depth: bool,
    metric_only_modes: Collection[str] = ("reasoning",),
    fallback: str = "direct",
) -> str:
    """Metric filter + free-weighted sampling (``fallback`` used only if pool empty)."""
    eligible = eligible_instruction_modes(
        modes,
        is_metric_depth=is_metric_depth,
        metric_only_modes=metric_only_modes,
    )
    if not eligible:
        return fallback
    return pick_instruction_mode(
        modes,
        is_metric_depth=is_metric_depth,
        metric_only_modes=metric_only_modes,
    )


def pick_stem_index(
    stems: Sequence[str],
    *,
    is_metric_depth: bool,
    metric_markers: Sequence[str] = RELATIVE_DISTANCE_STEM_METRIC_MARKERS,
) -> int:
    """Avoid metric-estimation stems when ``is_metric_depth`` is false."""
    eligible = [
        i
        for i, line in enumerate(stems)
        if is_metric_depth or not any(marker in line for marker in metric_markers)
    ]
    if not eligible:
        raise ValueError(
            f"no stem eligible for is_metric_depth={is_metric_depth!r} "
            f"(pool size {len(stems)})"
        )
    return rng().choice(eligible)
