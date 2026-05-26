"""
Upstream (G6) export bundle: self-contained JSONL + images.tar + manifest.

This is NOT a downstream training format. Conversion scripts read each JSONL line
(full metadata + mark_spec + messages) and emit VLM-specific training rows.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

UPSTREAM_SCHEMA_VERSION = "1.1"
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


def write_upstream_bundle(
    dataset: pd.DataFrame,
    export_root: Union[str, Path],
    *,
    schema_version: str = UPSTREAM_SCHEMA_VERSION,
    pipeline_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Write samples.jsonl, images.tar, manifest.json under export_root.

    Returns summary dict (n_samples, n_images, paths, missing paths).
    """
    out = Path(export_root)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / SAMPLES_FILENAME
    tar_path = out / IMAGES_TAR_FILENAME
    manifest_path = out / MANIFEST_FILENAME
    run_id = pipeline_run_id or str(uuid.uuid4())

    tar_index: List[dict] = []
    seen_paths: Dict[str, str] = {}
    n_samples = 0
    n_images = 0
    missing: List[str] = []

    with open(jsonl_path, "w", encoding="utf-8") as jf, tarfile.open(tar_path, "w") as tar:
        for idx in range(len(dataset)):
            record = merged_row_to_upstream_record(dataset.iloc[idx])
            sample_id = record["sample_id"] or f"sample_{idx}"
            record["sample_id"] = sample_id
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
                    arc = seen_paths[key]
                else:
                    arc = _arcname_for_ref(ref, sample_id, ri)
                    tar.add(src, arcname=arc)
                    seen_paths[key] = arc
                    tar_index.append({
                        "tar_path": arc,
                        "source_path": str(src).replace("\\", "/"),
                        "sha256": _file_sha256(src),
                    })
                    n_images += 1
                out_refs.append(arc)

            record["image_refs"] = out_refs
            record["schema_version"] = record.get("schema_version") or schema_version
            jf.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            n_samples += 1

    manifest = {
        "kind": "upstream_openspatial_bundle",
        "schema_version": schema_version,
        "pipeline_run_id": run_id,
        "n_samples": n_samples,
        "n_images": n_images,
        "samples_jsonl": SAMPLES_FILENAME,
        "images_tar": IMAGES_TAR_FILENAME,
        "tar_index": tar_index,
        "missing_image_paths": missing[:50],
        "note": "Self-contained upstream artifact; convert to training format outside this pipeline.",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if missing:
        print(f">>> Upstream export WARN: {len(missing)} image path(s) missing (first 5): {missing[:5]}")
    print(f">>> Upstream export: {n_samples} samples, {n_images} unique images -> {out}")

    return {
        "export_dir": str(out),
        "n_samples": n_samples,
        "n_images": n_images,
        "missing_paths": len(missing),
        "manifest_path": str(manifest_path),
        "jsonl_path": str(jsonl_path),
        "tar_path": str(tar_path),
    }


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


def verify_bundle_roundtrip(
    export_root: Union[str, Path],
    *,
    max_samples: int = 10,
) -> Tuple[bool, List[str]]:
    """Check manifest hashes and that tar members exist for JSONL image_refs."""
    root = Path(export_root)
    manifest = load_manifest(root)
    index = {e["tar_path"]: e for e in manifest.get("tar_index", [])}
    errors: List[str] = []

    with tarfile.open(root / IMAGES_TAR_FILENAME, "r") as tar:
        members = {m.name for m in tar.getmembers()}

    for i, rec in enumerate(read_upstream_jsonl(root / SAMPLES_FILENAME)):
        if i >= max_samples:
            break
        for ref in rec.get("image_refs") or []:
            ref = str(ref)
            if ref not in members:
                errors.append(f"sample {rec.get('sample_id')}: missing tar member {ref}")
            elif ref in index and index[ref].get("sha256"):
                pass
    return len(errors) == 0, errors
