"""
Upstream (G6) export: sharded JSONL + per-shard tar (+ dataset metadata.json).

Legacy monolithic layout (samples.jsonl + images.tar + manifest.json) is still
readable via :func:`read_upstream_bundle`; new exports use:

  {export_root}/jsonl/metadata_{shard:06d}.jsonl
  {export_root}/images/metadata_{shard:06d}.tar
  {export_root}/metadata.json

This is NOT a downstream training format. Conversion scripts read each JSONL line
(full metadata + mark_spec + messages) and emit VLM-specific training rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from dataset.export_stats import ExportStatsCollector

UPSTREAM_SCHEMA_VERSION = "1.1"
SHARD_SIZE = 8192
JSONL_SUBDIR = "jsonl"
IMAGES_SUBDIR = "images"
DATASET_METADATA_FILENAME = "metadata.json"
SHARD_BASENAME_FMT = "metadata_{:06d}"

# Legacy monolithic bundle (read-only)
MANIFEST_FILENAME = "manifest.json"
SAMPLES_FILENAME = "samples.jsonl"
IMAGES_TAR_FILENAME = "images.tar"


def _json_load(val: Any) -> Any:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        return json.loads(val)
    return val


def _json_safe(obj: Any) -> Any:
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return None
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def normalize_messages(raw: Any) -> List[dict]:
    """Aggregate parquet may store messages as flat list or [[turn messages]]."""
    msgs = _json_load(raw) if not isinstance(raw, list) else raw
    if not msgs:
        return []
    if len(msgs) == 1 and isinstance(msgs[0], list):
        inner = msgs[0]
        if inner and isinstance(inner[0], dict) and "from" in inner[0]:
            return inner
    if msgs and isinstance(msgs[0], dict) and "from" in msgs[0]:
        return msgs
    return msgs


def merged_row_to_upstream_record(row: Union[dict, Any]) -> dict:
    """Build one upstream JSONL record from aggregate merged_samples row."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    meta = _json_load(row.get("metadata_json")) or {}
    if not meta and isinstance(row.get("metadata"), dict):
        meta = row["metadata"]
    msgs = normalize_messages(row.get("messages_json") or row.get("messages"))
    refs = row.get("image_refs")
    if hasattr(refs, "tolist"):
        refs = refs.tolist()
    refs = [str(r) for r in (refs or []) if r]
    if not refs:
        refs = meta.get("image_refs") or []
        if hasattr(refs, "tolist"):
            refs = refs.tolist()
        refs = [str(r) for r in (refs or []) if r]

    return {
        "schema_version": row.get("schema_version") or UPSTREAM_SCHEMA_VERSION,
        "sample_id": str(row.get("sample_id") or ""),
        "merge_group_key": row.get("merge_group_key"),
        "messages": msgs,
        "metadata": _json_safe(meta),
        "image_refs": refs,
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _arcname_for_ref(ref: str, sample_id: str, index: int) -> str:
    ref = str(ref).replace("\\", "/")
    ext = Path(ref).suffix or ".jpg"
    if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    return f"images/{sample_id}/{index:02d}{ext.lower()}"


def shard_basename(shard_index: int) -> str:
    return SHARD_BASENAME_FMT.format(shard_index)


def is_sharded_upstream_root(export_root: Union[str, Path]) -> bool:
    return (Path(export_root) / JSONL_SUBDIR).is_dir()


def discover_shard_pairs(export_root: Union[str, Path]) -> List[Tuple[Path, Optional[Path]]]:
    """Return (jsonl_path, tar_path) pairs sorted by shard index."""
    root = Path(export_root)
    jsonl_dir = root / JSONL_SUBDIR
    images_dir = root / IMAGES_SUBDIR
    if not jsonl_dir.is_dir():
        return []
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for jf in sorted(jsonl_dir.glob("metadata_*.jsonl")):
        stem = jf.stem
        tar = images_dir / f"{stem}.tar"
        pairs.append((jf, tar if tar.is_file() else None))
    return pairs


def read_sharded_upstream(export_root: Union[str, Path]) -> List[dict]:
    """Load all JSONL records; attach ``_bundle_root`` and ``_shard_tar`` for image I/O."""
    root = Path(export_root)
    records: List[dict] = []
    for jf, tar in discover_shard_pairs(root):
        for rec in read_upstream_jsonl(jf):
            rec = dict(rec)
            rec["_bundle_root"] = str(root)
            if tar is not None:
                rec["_shard_tar"] = str(tar)
            records.append(rec)
    return records


def _pack_record_images(
    record: dict,
    *,
    tar: tarfile.TarFile,
    seen_paths: Dict[str, str],
    stats: ExportStatsCollector,
) -> Tuple[dict, int, List[str]]:
    """Rewrite image_refs to tar member paths; return (record, n_new_images, missing)."""
    missing: List[str] = []
    n_new = 0
    sample_id = record["sample_id"] or "sample"
    src_refs = list(record.get("image_refs") or [])
    out_refs: List[str] = []

    for ri, ref in enumerate(src_refs):
        src = Path(ref)
        if not src.is_file():
            missing.append(ref)
            out_refs.append(ref)
            continue
        key = str(src.resolve())
        if key in seen_paths:
            out_refs.append(seen_paths[key])
            continue
        arc = _arcname_for_ref(ref, sample_id, ri)
        tar.add(src, arcname=arc)
        seen_paths[key] = arc
        stats.observe_resolution_file(str(src))
        n_new += 1
        out_refs.append(arc)

    record = dict(record)
    record["image_refs"] = out_refs
    return record, n_new, missing


def write_sharded_upstream_bundle(
    dataset: pd.DataFrame,
    export_root: Union[str, Path],
    *,
    schema_version: str = UPSTREAM_SCHEMA_VERSION,
    pipeline_run_id: Optional[str] = None,
    view_scope: str = "singleview",
    shard_size: int = SHARD_SIZE,
) -> Dict[str, Any]:
    """
    Write sharded jsonl/tar under export_root and dataset-level metadata.json.

    Returns summary dict (paths, counts).
    """
    out = Path(export_root)
    jsonl_dir = out / JSONL_SUBDIR
    images_dir = out / IMAGES_SUBDIR
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    run_id = pipeline_run_id or str(uuid.uuid4())

    stats = ExportStatsCollector(view_scope=view_scope)
    shard_idx = 0
    buffer: List[dict] = []
    n_images_packed = 0
    all_missing: List[str] = []
    shard_summaries: List[dict] = []
    total_records = len(dataset)
    records_processed = 0
    progress_every = 500

    def flush_shard() -> None:
        nonlocal shard_idx, n_images_packed, records_processed
        if not buffer:
            return
        base = shard_basename(shard_idx)
        jsonl_path = jsonl_dir / f"{base}.jsonl"
        tar_path = images_dir / f"{base}.tar"
        seen_paths: Dict[str, str] = {}
        shard_missing: List[str] = []
        shard_images = 0

        print(
            f">>> Upstream export shard {shard_idx + 1}: "
            f"writing {len(buffer)} sample(s) to {tar_path.name}"
        )
        with open(jsonl_path, "w", encoding="utf-8") as jf, tarfile.open(tar_path, "w") as tar:
            for i, record in enumerate(buffer, start=1):
                record, n_new, missing = _pack_record_images(
                    record, tar=tar, seen_paths=seen_paths, stats=stats
                )
                shard_missing.extend(missing)
                shard_images += n_new
                record["schema_version"] = record.get("schema_version") or schema_version
                stats.observe_record(record)
                jf.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                records_processed += 1
                if i % progress_every == 0 or i == len(buffer):
                    print(
                        f">>> Upstream export progress: shard {shard_idx + 1} "
                        f"{i}/{len(buffer)} samples, "
                        f"global {records_processed}/{total_records}"
                    )

        n_images_packed += shard_images
        all_missing.extend(shard_missing)
        shard_summaries.append({
            "shard": shard_idx,
            "basename": base,
            "n_samples": len(buffer),
            "n_images": shard_images,
            "jsonl": str(jsonl_path.relative_to(out)).replace("\\", "/"),
            "tar": str(tar_path.relative_to(out)).replace("\\", "/"),
        })
        print(
            f">>> Upstream export shard {shard_idx + 1} done: "
            f"{len(buffer)} sample(s), {shard_images} image(s)"
        )
        shard_idx += 1
        buffer.clear()

    for idx in range(len(dataset)):
        record = merged_row_to_upstream_record(dataset.iloc[idx])
        record["sample_id"] = record["sample_id"] or f"sample_{idx}"
        buffer.append(record)
        if len(buffer) >= shard_size:
            flush_shard()
    flush_shard()

    meta_path = out / DATASET_METADATA_FILENAME
    dataset_meta = stats.finalize(
        schema_version=schema_version,
        pipeline_run_id=run_id,
        shard_size=shard_size,
        n_shards=shard_idx,
        n_images_packed=n_images_packed,
        missing_paths=len(all_missing),
    )
    dataset_meta["shards"] = shard_summaries
    meta_path.write_text(
        json.dumps(dataset_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if all_missing:
        print(
            f">>> Upstream export WARN: {len(all_missing)} image path(s) missing "
            f"(first 5): {all_missing[:5]}"
        )
    print(
        f">>> Upstream export: {stats.n_samples} samples, {n_images_packed} images, "
        f"{shard_idx} shard(s) -> {out}"
    )

    return {
        "export_dir": str(out),
        "n_samples": stats.n_samples,
        "n_images": n_images_packed,
        "n_shards": shard_idx,
        "missing_paths": len(all_missing),
        "metadata_path": str(meta_path),
        "jsonl_dir": str(jsonl_dir),
        "images_dir": str(images_dir),
    }


def write_upstream_bundle(
    dataset: pd.DataFrame,
    export_root: Union[str, Path],
    *,
    schema_version: str = UPSTREAM_SCHEMA_VERSION,
    pipeline_run_id: Optional[str] = None,
    view_scope: str = "singleview",
) -> Dict[str, Any]:
    """Write sharded upstream bundle (alias for :func:`write_sharded_upstream_bundle`)."""
    return write_sharded_upstream_bundle(
        dataset,
        export_root,
        schema_version=schema_version,
        pipeline_run_id=pipeline_run_id,
        view_scope=view_scope,
    )


def read_upstream_bundle(export_root: Union[str, Path]) -> List[dict]:
    """Load records from sharded or legacy monolithic export root."""
    root = Path(export_root)
    if is_sharded_upstream_root(root):
        return read_sharded_upstream(root)
    jsonl_path = root / SAMPLES_FILENAME
    if jsonl_path.is_file():
        return read_upstream_jsonl(jsonl_path)
    raise FileNotFoundError(
        f"No upstream bundle under {root} (expected {JSONL_SUBDIR}/ or {SAMPLES_FILENAME})"
    )


def read_upstream_jsonl(path: Union[str, Path]) -> List[dict]:
    records: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_manifest(export_root: Union[str, Path]) -> dict:
    p = Path(export_root) / MANIFEST_FILENAME
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_bundle_path(export_root: Union[str, Path], image_ref: str) -> Path:
    """Map JSONL image_ref (tar member path) to extracted or tar-adjacent logical path."""
    root = Path(export_root)
    ref = str(image_ref).replace("\\", "/")
    if ref.startswith("images/"):
        return root / ref
    return root / ref


def open_shard_tar(shard_tar: Union[str, Path]) -> tarfile.TarFile:
    return tarfile.open(Path(shard_tar), "r")


def resolve_shard_image(
    image_ref: str,
    *,
    bundle_root: Optional[str] = None,
    shard_tar: Optional[str] = None,
    tar_cache: Optional[dict] = None,
) -> Optional[bytes]:
    """Read image bytes from a shard tar (or legacy monolithic images.tar)."""
    ref = str(image_ref).replace("\\", "/")
    if bundle_root:
        local = Path(bundle_root) / ref
        if local.is_file():
            return local.read_bytes()

    cache = tar_cache if tar_cache is not None else {}
    tar_path = shard_tar
    if not tar_path and bundle_root:
        legacy = Path(bundle_root) / IMAGES_TAR_FILENAME
        if legacy.is_file():
            tar_path = str(legacy)
    if not tar_path:
        return None

    if cache.get("tar_path") != tar_path:
        if "tar" in cache:
            cache["tar"].close()
        cache["tar_path"] = tar_path
        cache["tar"] = tarfile.open(tar_path, "r")
        cache["members"] = {m.name for m in cache["tar"].getmembers()}

    if ref not in cache.get("members", set()):
        return None
    extracted = cache["tar"].extractfile(ref)
    return extracted.read() if extracted else None


def verify_bundle_roundtrip(
    export_root: Union[str, Path],
    *,
    max_samples: int = 10,
) -> Tuple[bool, List[str]]:
    """Check that each JSONL image_ref exists in its paired shard tar (or legacy bundle)."""
    root = Path(export_root)
    errors: List[str] = []
    checked = 0

    if is_sharded_upstream_root(root):
        for jf, tar_path in discover_shard_pairs(root):
            if tar_path is None:
                errors.append(f"shard {jf.name}: missing tar")
                continue
            with tarfile.open(tar_path, "r") as tar:
                members = {m.name for m in tar.getmembers()}
            for rec in read_upstream_jsonl(jf):
                if checked >= max_samples:
                    return len(errors) == 0, errors
                for ref in rec.get("image_refs") or []:
                    ref = str(ref)
                    if ref.startswith("images/") and ref not in members:
                        errors.append(
                            f"sample {rec.get('sample_id')}: missing tar member {ref} in {tar_path.name}"
                        )
                checked += 1
        return len(errors) == 0, errors

    if (root / MANIFEST_FILENAME).is_file() and (root / SAMPLES_FILENAME).is_file():
        with tarfile.open(root / IMAGES_TAR_FILENAME, "r") as tar:
            members = {m.name for m in tar.getmembers()}
        for i, rec in enumerate(read_upstream_jsonl(root / SAMPLES_FILENAME)):
            if i >= max_samples:
                break
            for ref in rec.get("image_refs") or []:
                ref = str(ref)
                if ref not in members:
                    errors.append(f"sample {rec.get('sample_id')}: missing tar member {ref}")
        return len(errors) == 0, errors

    errors.append(f"not a recognized upstream bundle: {root}")
    return False, errors
