#!/usr/bin/env python3
"""
Convert OpenSpatial upstream export bundles to Pangu ML training format.

Input (upstream export):
  {export_dir}/jsonl/metadata_{shard:06d}.jsonl
  {export_dir}/images/metadata_{shard:06d}.tar

Output (Pangu ML, see HW_pangu_ml/docs/pangu_ml_data_schema.md):
  {output_root}/jsonl/data_{shard:06d}.jsonl
  {output_root}/images/data_{shard:06d}.tar

Sharding: 1:1 with upstream (metadata_{N} -> data_{N}). Use ``--num-workers``
for shard-level parallel conversion (ProcessPoolExecutor).

Rules (project-specific, stricter than schema 4.2):
  - All images (with every mark overlay) appear only in the first user turn, before text.
  - Later user turns contain text only.
  - Image placeholder tokens (<image>, <|image|>, etc.) are stripped from all text.
  - Q/A prose: in-place replace legacy ``tag-(red box)`` in messages with
    ``tag (<phrase>)``; box/point pick randomly from short phrase candidates.

Sample id: export ``sample_id`` (sanitized for tar paths).
Image tar paths: ``{safe_sample_id}_{view_index:02d}.jpg`` (or .png).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Avoid task.annotation.core.__init__ (pulls open3d via BaseAnnotationTask).
import types  # noqa: E402

_CORE_PKG = "task.annotation.core"
_CORE_DIR = _REPO_ROOT / "task" / "annotation" / "core"
for _pkg in ("task", "task.annotation", _CORE_PKG):
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        if _pkg == _CORE_PKG:
            _mod.__path__ = [str(_CORE_DIR)]
        sys.modules[_pkg] = _mod

from dataset.upstream_export import (  # noqa: E402
    discover_shard_pairs,
    normalize_messages,
    resolve_shard_image,
)
from task.annotation.core.mark_spec import (  # noqa: E402
    MARK_SPEC_VERSION,
    all_slots_flat,
    render_mark,
    slots_for_view,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

ROLE_MAP = {"human": "user", "gpt": "assistant"}
MIME_PNG = "image/png"
MIME_JPEG = "image/jpeg"
_JPEG_SOI = b"\xff\xd8"
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_PASSTHROUGH_MODES = frozenset({"RGB", "RGBA", "L", "1"})
IMAGE_TOKEN_RE = re.compile(
    r"\s*(?:<\|image(?:_pad)?\|>|<image(?:_pad)?(?:\s[^>]*)?>"
    r")\s*",
    flags=re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")
PATH_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
PANGU_SHARD_BASENAME_FMT = "data_{:06d}"

# Legacy viz form baked into exported messages: chair-(red box)
LEGACY_MARK_SUFFIX_RE = re.compile(
    r"\b(\w+)-\(\s*(\w+)\s+(box|mask|point)\s*\)",
    flags=re.IGNORECASE,
)

# Short parenthetical phrases; ``{color}`` filled at replace time.
BOX_MARK_PHRASES = (
    "in the {color} box",
    "inside the {color} box",
    "in a {color} box",
    "within the {color} box",
    "highlighted by a {color} box",
    "marked with a {color} box",
)
POINT_MARK_PHRASES = (
    "at the {color} point",
    "on the {color} point",
    "at a {color} point",
    "by the {color} point",
    "highlighted by a {color} point",
    "marked with a {color} point",
)


def pick_mark_phrase_for_kind(color: str, kind: str) -> str:
    """Random short phrase for box/point; mask keeps a single template."""
    color = str(color or "").strip().lower()
    kind = str(kind or "").strip().lower()
    if kind == "mask":
        return f"highlighted by a {color} mask"
    if kind == "point":
        template = random.choice(POINT_MARK_PHRASES)
    else:
        template = random.choice(BOX_MARK_PHRASES)
    return template.format(color=color)


def convert_legacy_marks_to_natural(text: str) -> str:
    """
    In-place: ``chair-(red box)`` -> ``chair (in the red box)`` (phrase varies).

    Tag, color, and kind are taken from the legacy token — no second pass over bare tags.
    """

    def _repl(match: re.Match) -> str:
        tag, color, kind = match.group(1), match.group(2), match.group(3)
        return f"{tag} ({pick_mark_phrase_for_kind(color, kind)})"

    out = LEGACY_MARK_SUFFIX_RE.sub(_repl, text or "")
    return re.sub(r"  +", " ", out).strip()


def count_legacy_mark_tokens(text: str) -> int:
    return len(LEGACY_MARK_SUFFIX_RE.findall(text or ""))


def count_legacy_marks_in_pairs(pairs: List[Tuple[str, str]]) -> int:
    return sum(
        count_legacy_mark_tokens(q) + count_legacy_mark_tokens(a)
        for q, a in pairs
    )


def slot_has_renderable_geometry(slot: dict) -> bool:
    geom = slot.get("geometry") or {}
    kind = str(slot.get("mark_kind") or "box").lower()
    if kind == "point":
        return bool(geom.get("uv"))
    if kind == "mask":
        return bool(geom.get("mask_ref") or geom.get("box_2d") or geom.get("uv"))
    return bool(geom.get("box_2d") or geom.get("uv"))


def count_renderable_slots(mark_spec: Optional[dict]) -> int:
    if not mark_spec:
        return 0
    return sum(
        1 for s in all_slots_flat(mark_spec)
        if isinstance(s, dict) and slot_has_renderable_geometry(s)
    )


@dataclass
class ConvertStats:
    total_seen: int = 0
    converted: int = 0
    skipped: int = 0
    skip_reasons: Counter = field(default_factory=Counter)
    legacy_mark_tokens: int = 0
    renderable_slots: int = 0
    views_without_render: int = 0
    mark_text_render_mismatch: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenSpatial upstream export to Pangu ML format."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Upstream export root (jsonl/metadata_*.jsonl + images/metadata_*.tar).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root; creates jsonl/ and images/ subdirectories.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Shard-level parallel workers (default: min(shard_count, cpu_count)).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick validation runs (forces sequential).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for mark phrase randomization (per-shard: seed + shard_index).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return parser.parse_args()


def strip_image_placeholder_tokens(text: str) -> str:
    cleaned = IMAGE_TOKEN_RE.sub(" ", text or "")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def sanitize_sample_id(sample_id: str, fallback_index: int) -> str:
    raw = (sample_id or "").strip() or f"sample_{fallback_index}"
    safe = PATH_SAFE_RE.sub("_", raw).strip("._-")
    return safe[:200] or f"sample_{fallback_index}"


def to_text_content(text: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": {"type": "string", "format": "utf-8", "string": text},
    }


def to_image_content(
    relative_path: str,
    width: int,
    height: int,
    mime: str,
) -> Dict[str, Any]:
    return {
        "type": "image",
        "image": {
            "type": "relative_path",
            "format": mime,
            "relative_path": relative_path,
            "width": int(width),
            "height": int(height),
        },
    }


def _prepare_for_png_save(img: Any) -> Any:
    if img.mode in _PNG_PASSTHROUGH_MODES:
        return img
    return img.convert("RGB")


def encode_image_for_pangu_tar(image_bytes: bytes) -> Tuple[bytes, int, int, str]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.width, img.height
            fmt = (img.format or "").upper()

            if fmt == "JPEG" and image_bytes.startswith(_JPEG_SOI):
                return image_bytes, width, height, MIME_JPEG

            if (
                fmt == "PNG"
                and image_bytes.startswith(_PNG_SIG)
                and img.mode in _PNG_PASSTHROUGH_MODES
            ):
                return image_bytes, width, height, MIME_PNG

            if fmt == "PNG":
                im_out = _prepare_for_png_save(img)
                buf = io.BytesIO()
                im_out.save(buf, format="PNG", optimize=False)
                return buf.getvalue(), width, height, MIME_PNG

            im_j = img
            if im_j.mode in ("RGBA", "LA", "P"):
                im_j = im_j.convert("RGB")
            elif im_j.mode != "RGB":
                im_j = im_j.convert("RGB")
            buf = io.BytesIO()
            im_j.save(buf, format="JPEG", quality=95)
            return buf.getvalue(), width, height, MIME_JPEG
    except Exception:
        mime = MIME_PNG if image_bytes.startswith(_PNG_SIG) else MIME_JPEG
        return image_bytes, -1, -1, mime


def build_tar_relative_path(sample_id: str, img_idx: int, mime: str) -> str:
    ext = ".png" if mime == MIME_PNG else ".jpg"
    return f"{sample_id}_{img_idx:02d}{ext}"


def parse_qa_pairs(messages: List[dict]) -> Optional[List[Tuple[str, str]]]:
    if not messages:
        return None
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(messages), 2):
        if i + 1 >= len(messages):
            return None
        human = messages[i]
        gpt = messages[i + 1]
        if not isinstance(human, dict) or not isinstance(gpt, dict):
            return None
        if str(human.get("from", "")).strip().lower() != "human":
            return None
        if str(gpt.get("from", "")).strip().lower() != "gpt":
            return None
        q = "" if human.get("value") is None else str(human.get("value"))
        a = "" if gpt.get("value") is None else str(gpt.get("value"))
        pairs.append((q, a))
    return pairs if pairs else None


def turn_mark_specs(metadata: dict) -> List[Optional[dict]]:
    turns = metadata.get("turns") or []
    specs: List[Optional[dict]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            specs.append(None)
            continue
        ms = turn.get("mark_spec")
        specs.append(ms if isinstance(ms, dict) else None)
    return specs


def mark_spec_for_turn(
    turn_specs: List[Optional[dict]],
    turn_index: int,
    sample_mark_spec: Optional[dict],
) -> Optional[dict]:
    if turn_index < len(turn_specs) and turn_specs[turn_index]:
        return turn_specs[turn_index]
    return sample_mark_spec


def flat_mark_spec_for_view(
    mark_spec: Optional[dict],
    view_index: int,
) -> Optional[dict]:
    """Return a render-ready spec ``{version, mark_kinds, slots}`` for one QA image."""
    if not mark_spec or not isinstance(mark_spec, dict):
        return None
    slots = slots_for_view(mark_spec, view_index)
    if not slots:
        return None
    kinds = sorted({s.get("mark_kind") for s in slots if s.get("mark_kind")})
    return {"version": MARK_SPEC_VERSION, "mark_kinds": kinds, "slots": slots}


def combined_mark_spec_for_view(
    turn_specs: List[Optional[dict]],
    sample_mark_spec: Optional[dict],
    view_index: int,
) -> Optional[dict]:
    """Union all mark slots on one view across turns (flat spec for render_mark)."""
    seen: set = set()
    merged_slots: List[dict] = []

    sources: List[dict] = []
    for ms in turn_specs:
        if ms:
            sources.append(ms)
    if sample_mark_spec:
        sources.append(sample_mark_spec)

    for ms in sources:
        flat = flat_mark_spec_for_view(ms, view_index)
        if not flat:
            continue
        for slot in flat.get("slots") or []:
            key = (slot.get("slot_id"), slot.get("tag"), slot.get("mark_kind"))
            if key in seen:
                continue
            seen.add(key)
            merged_slots.append(slot)

    if not merged_slots:
        return None
    kinds = sorted({s.get("mark_kind") for s in merged_slots if s.get("mark_kind")})
    return {"version": MARK_SPEC_VERSION, "mark_kinds": kinds, "slots": merged_slots}


def count_renderable_slots_for_view(
    turn_specs: List[Optional[dict]],
    sample_mark_spec: Optional[dict],
    view_index: int,
) -> int:
    flat = combined_mark_spec_for_view(turn_specs, sample_mark_spec, view_index)
    if not flat:
        return 0
    return sum(
        1 for s in flat.get("slots") or []
        if isinstance(s, dict) and slot_has_renderable_geometry(s)
    )


def audit_sample_marks(
    record: dict,
    pairs: List[Tuple[str, str]],
) -> Tuple[int, int, int, int]:
    """
    Returns (legacy_tokens, renderable_slots, views_without_render, mismatch_flag).
    """
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    sample_mark_spec = metadata.get("mark_spec")
    if sample_mark_spec is not None and not isinstance(sample_mark_spec, dict):
        sample_mark_spec = None
    turn_specs = turn_mark_specs(metadata)

    legacy = count_legacy_marks_in_pairs(pairs)
    refs = list(record.get("image_refs") or [])
    renderable = 0
    views_no_render = 0
    for vi in range(len(refs)):
        n = count_renderable_slots_for_view(turn_specs, sample_mark_spec, vi)
        renderable += n
        flat = combined_mark_spec_for_view(turn_specs, sample_mark_spec, vi)
        spec_slots = len(flat.get("slots") or []) if flat else 0
        if spec_slots > 0 and n == 0:
            views_no_render += 1

    mismatch = int(legacy > 0 and renderable < legacy)
    return legacy, renderable, views_no_render, mismatch


def apply_marked_text(text: str, _mark_spec: Optional[dict] = None) -> str:
    stripped = strip_image_placeholder_tokens(text)
    if not stripped:
        return ""
    return convert_legacy_marks_to_natural(stripped)


def render_marked_image_bytes(
    raw_bytes: bytes,
    mark_spec: Optional[dict],
    *,
    view_index: int,
) -> bytes:
    from PIL import Image

    img = Image.open(io.BytesIO(raw_bytes))
    flat = flat_mark_spec_for_view(mark_spec, view_index) if mark_spec else None
    if flat and flat.get("slots"):
        out = render_mark(img, flat, view_index=0)
        return out["bytes"]
    return encode_image_for_pangu_tar(raw_bytes)[0]


def load_and_encode_marked_images(
    record: dict,
    safe_id: str,
    turn_specs: List[Optional[dict]],
    sample_mark_spec: Optional[dict],
    tar_cache: dict,
) -> Tuple[List[Tuple[str, bytes, int, int, str]], List[str]]:
    """
    Returns (image_rows, errors).
    image_rows: (tar_relative_path, encoded_bytes, width, height, mime)
    """
    refs = list(record.get("image_refs") or [])
    bundle_root = record.get("_bundle_root")
    shard_tar = record.get("_shard_tar")
    errors: List[str] = []
    rows: List[Tuple[str, bytes, int, int, str]] = []

    for vi, ref in enumerate(refs):
        raw = resolve_shard_image(
            str(ref),
            bundle_root=bundle_root,
            shard_tar=shard_tar,
            tar_cache=tar_cache,
        )
        if not raw:
            errors.append(f"missing image: {ref}")
            continue

        view_ms = combined_mark_spec_for_view(turn_specs, sample_mark_spec, vi)
        try:
            marked_bytes = render_marked_image_bytes(
                raw, view_ms, view_index=vi,
            )
            encoded, width, height, mime = encode_image_for_pangu_tar(marked_bytes)
            rel = build_tar_relative_path(safe_id, vi, mime)
            rows.append((rel, encoded, width, height, mime))
        except Exception as exc:
            errors.append(f"render failed {ref}: {exc}")

    return rows, errors


def build_pangu_sample(
    record: dict,
    row_index: int,
    tar_cache: dict,
) -> Tuple[Optional[Dict[str, Any]], List[Tuple[str, bytes]], Optional[str]]:
    messages = normalize_messages(record.get("messages"))
    pairs = parse_qa_pairs(messages)
    if not pairs:
        return None, [], "invalid_messages"

    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    sample_mark_spec = metadata.get("mark_spec")
    if sample_mark_spec is not None and not isinstance(sample_mark_spec, dict):
        sample_mark_spec = None

    turn_specs = turn_mark_specs(metadata)
    sample_id = sanitize_sample_id(str(record.get("sample_id") or ""), row_index)

    image_rows, img_errors = load_and_encode_marked_images(
        record, sample_id, turn_specs, sample_mark_spec, tar_cache,
    )
    if img_errors and not image_rows:
        return None, [], "missing_images"

    data: List[Dict[str, Any]] = []

    first_q, first_a = pairs[0]
    ms0 = mark_spec_for_turn(turn_specs, 0, sample_mark_spec)
    first_user_content: List[Dict[str, Any]] = []
    for rel, _b, w, h, mime in image_rows:
        first_user_content.append(to_image_content(rel, w, h, mime))
    q0 = apply_marked_text(first_q, ms0)
    if q0:
        first_user_content.append(to_text_content(q0))
    if not first_user_content:
        return None, [], "empty_first_user"

    data.append({"role": "user", "content": first_user_content})

    a0 = apply_marked_text(first_a, ms0)
    if not a0:
        return None, [], "empty_first_answer"
    data.append({"role": "assistant", "content": [to_text_content(a0)]})

    for ti in range(1, len(pairs)):
        q_raw, a_raw = pairs[ti]
        ms = mark_spec_for_turn(turn_specs, ti, sample_mark_spec)
        q = apply_marked_text(q_raw, ms)
        a = apply_marked_text(a_raw, ms)
        if not q:
            return None, [], f"empty_question_turn_{ti}"
        if not a:
            return None, [], f"empty_answer_turn_{ti}"
        data.append({"role": "user", "content": [to_text_content(q)]})
        data.append({"role": "assistant", "content": [to_text_content(a)]})

    sample = {
        "meta_prompt": [""],
        "data": data,
        "id": str(record.get("sample_id") or sample_id),
    }
    tar_members = [(rel, blob) for rel, blob, _w, _h, _m in image_rows]
    skip_reason = None
    if img_errors:
        skip_reason = "partial_images"
    return sample, tar_members, skip_reason


def shard_index_from_jsonl(jf_path: Path) -> int:
    return int(jf_path.stem.rsplit("_", 1)[-1])


@dataclass
class ShardJob:
    shard_index: int
    jsonl_path: str
    tar_path: Optional[str]
    export_dir: str
    output_root: str
    global_index_start: int
    max_converted: Optional[int] = None
    seed: Optional[int] = None


def _merge_stats(into: ConvertStats, other: ConvertStats) -> None:
    into.total_seen += other.total_seen
    into.converted += other.converted
    into.skipped += other.skipped
    into.skip_reasons.update(other.skip_reasons)
    into.legacy_mark_tokens += other.legacy_mark_tokens
    into.renderable_slots += other.renderable_slots
    into.views_without_render += other.views_without_render
    into.mark_text_render_mismatch += other.mark_text_render_mismatch


def write_shard(
    shard_index: int,
    samples: List[Dict[str, Any]],
    tar_members_by_sample: List[List[Tuple[str, bytes]]],
    output_root: Path,
) -> None:
    jsonl_dir = output_root / "jsonl"
    images_dir = output_root / "images"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    base = PANGU_SHARD_BASENAME_FMT.format(shard_index)
    jsonl_path = jsonl_dir / f"{base}.jsonl"
    tar_path = images_dir / f"{base}.tar"

    with open(jsonl_path, "w", encoding="utf-8") as jf, tarfile.open(tar_path, "w") as tar:
        for sample, members in zip(samples, tar_members_by_sample):
            jf.write(json.dumps(sample, ensure_ascii=False) + "\n")
            for rel_path, blob in members:
                info = tarfile.TarInfo(name=rel_path)
                info.size = len(blob)
                tar.addfile(tarinfo=info, fileobj=io.BytesIO(blob))


def convert_one_shard(job: ShardJob) -> ConvertStats:
    """Convert a single upstream shard to one Pangu ML shard (1:1 id)."""
    stats = ConvertStats()
    if job.seed is not None:
        random.seed(job.seed + job.shard_index)

    jf_path = Path(job.jsonl_path)
    tar_path = Path(job.tar_path) if job.tar_path else None
    output_root = Path(job.output_root)
    bundle_root = job.export_dir
    shard_tar = str(tar_path) if tar_path and tar_path.is_file() else None

    buffer_samples: List[Dict[str, Any]] = []
    buffer_members: List[List[Tuple[str, bytes]]] = []
    tar_cache: dict = {}
    line_index = 0

    with jf_path.open("r", encoding="utf-8") as jf:
        for line in jf:
            if job.max_converted is not None and stats.converted >= job.max_converted:
                break

            line = line.strip()
            if not line:
                continue

            stats.total_seen += 1
            record = json.loads(line)
            record["_bundle_root"] = bundle_root
            if shard_tar:
                record["_shard_tar"] = shard_tar

            global_index = job.global_index_start + line_index
            line_index += 1

            sample, members, partial = build_pangu_sample(
                record, global_index, tar_cache,
            )

            if sample is None:
                stats.skipped += 1
                stats.skip_reasons[partial or "unknown"] += 1
                continue

            pairs = parse_qa_pairs(normalize_messages(record.get("messages")))
            if pairs:
                legacy, renderable, views_no, mismatch = audit_sample_marks(
                    record, pairs,
                )
                stats.legacy_mark_tokens += legacy
                stats.renderable_slots += renderable
                stats.views_without_render += views_no
                stats.mark_text_render_mismatch += mismatch

            stats.converted += 1
            if partial:
                stats.skip_reasons[partial] += 1
            buffer_samples.append(sample)
            buffer_members.append(members)

    if "tar" in tar_cache:
        tar_cache["tar"].close()

    if buffer_samples:
        write_shard(job.shard_index, buffer_samples, buffer_members, output_root)

    return stats


def _build_shard_jobs(
    pairs: List[Tuple[Path, Optional[Path]]],
    export_dir: Path,
    output_root: Path,
    *,
    seed: Optional[int],
) -> List[ShardJob]:
    jobs: List[ShardJob] = []

    for jf_path, tar_path in pairs:
        shard_idx = shard_index_from_jsonl(jf_path)
        jobs.append(ShardJob(
            shard_index=shard_idx,
            jsonl_path=str(jf_path),
            tar_path=str(tar_path) if tar_path else None,
            export_dir=str(export_dir),
            output_root=str(output_root),
            global_index_start=shard_idx * 10_000_000,
            max_converted=None,
            seed=seed,
        ))

    return jobs


def convert_export(
    export_dir: Path,
    output_root: Path,
    *,
    num_workers: Optional[int] = None,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    show_progress: bool = True,
) -> ConvertStats:
    stats = ConvertStats()
    pairs = discover_shard_pairs(export_dir)
    if not pairs:
        raise FileNotFoundError(f"No upstream shards under {export_dir / 'jsonl'}")

    jobs = _build_shard_jobs(pairs, export_dir, output_root, seed=seed)
    if not jobs:
        return stats

    if max_samples is not None:
        workers = 1
    else:
        cpu = os.cpu_count() or 1
        workers = num_workers if num_workers is not None else min(len(jobs), cpu)
        workers = max(1, min(workers, len(jobs)))

    print(
        f">>> Converting {len(jobs)} shard(s) with {workers} worker(s) (1:1 shard id)",
        flush=True,
    )

    if workers == 1:
        remaining = max_samples
        job_iter: Any = jobs
        if tqdm is not None and show_progress:
            job_iter = tqdm(jobs, desc="shards", unit="shard")
        for job in job_iter:
            if remaining is not None and remaining <= 0:
                break
            job.max_converted = remaining
            shard_stats = convert_one_shard(job)
            _merge_stats(stats, shard_stats)
            if remaining is not None:
                remaining -= shard_stats.converted
        return stats

    if tqdm is not None and show_progress:
        pbar = tqdm(total=len(jobs), desc="shards", unit="shard")
    else:
        pbar = None

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(convert_one_shard, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            shard_stats = fut.result()
            _merge_stats(stats, shard_stats)
            if pbar is not None:
                pbar.update(1)
            else:
                print(
                    f">>> shard {job.shard_index}: {shard_stats.converted} converted, "
                    f"{shard_stats.skipped} skipped",
                    flush=True,
                )

    if pbar is not None:
        pbar.close()

    return stats


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    output_root = args.output_root.resolve()

    if not export_dir.is_dir():
        raise SystemExit(f"export-dir not found: {export_dir}")

    print(f">>> Pangu ML convert: {export_dir} -> {output_root}", flush=True)

    if args.max_samples is not None and args.num_workers not in (None, 1):
        print(">>> max_samples set; running sequentially (num_workers=1)")

    stats = convert_export(
        export_dir,
        output_root,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
        show_progress=not args.no_progress,
    )

    print(
        f">>> Pangu ML export: {stats.converted} converted, "
        f"{stats.skipped} skipped (of {stats.total_seen} seen) -> {output_root}"
    )
    if stats.converted:
        print(
            f">>> Mark audit: legacy_tokens={stats.legacy_mark_tokens}, "
            f"renderable_slots={stats.renderable_slots}, "
            f"views_without_render={stats.views_without_render}, "
            f"text_render_mismatch_samples={stats.mark_text_render_mismatch}",
            flush=True,
        )
        if stats.mark_text_render_mismatch:
            print(
                ">>> WARNING: some samples mention more legacy mark tokens in text "
                "than renderable slots in mark_spec (marks missing in tar images).",
                flush=True,
            )
    if stats.skip_reasons:
        print(">>> Skip reasons:", dict(stats.skip_reasons))


if __name__ == "__main__":
    main()
