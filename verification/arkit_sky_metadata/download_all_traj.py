"""Download ARKitScenes lowres_wide.traj only into verification data dir."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARKIT_BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
SPLITS_URL = f"{ARKIT_BASE}/raw/raw_train_val_splits.csv"
METADATA_URL = f"{ARKIT_BASE}/raw/metadata.csv"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
LOCAL_SPLITS = REPO_ROOT / "tools" / "ARKitScenes" / "raw" / "raw_train_val_splits.csv"


def _download_url_curl(url: str, dest: Path) -> bool:
    """Use curl like official download_data.py (urllib often gets 403)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.is_file():
        tmp.unlink(missing_ok=True)
    cmd = ["curl", "-f", "-L", url, "-o", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, cwd=str(dest.parent))
    except subprocess.CalledProcessError:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    return True


def _ensure_csv(url: str, dest: Path, local_fallback: Path | None = None) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    if local_fallback is not None and local_fallback.is_file():
        print(f"Copying {local_fallback.name} -> {dest}")
        shutil.copy2(local_fallback, dest)
        return
    print(f"Downloading {dest.name} ...")
    if not _download_url_curl(url, dest):
        raise RuntimeError(f"Failed to download {url}")


def _load_splits(splits_path: Path, split_filter: str | None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(splits_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold = row["fold"].strip()
            if split_filter and fold != split_filter:
                continue
            vid = str(int(float(row["video_id"])))
            rows.append((fold, vid))
    return rows


def _traj_dest(data_dir: Path, fold: str, video_id: str) -> Path:
    return data_dir / "traj" / fold / video_id / "lowres_wide.traj"


def _download_one(data_dir: Path, fold: str, video_id: str) -> tuple[str, str, str]:
    """Return (video_id, status, detail). status: ok | missing | error"""
    dest = _traj_dest(data_dir, fold, video_id)
    if dest.is_file() and dest.stat().st_size > 0:
        return video_id, "ok", "cached"
    url = f"{ARKIT_BASE}/raw/{fold}/{video_id}/lowres_wide.traj"
    try:
        if _download_url_curl(url, dest):
            return video_id, "ok", "downloaded"
        return video_id, "missing", "404"
    except Exception as exc:
        return video_id, "error", str(exc)[:200]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download all ARKitScenes lowres_wide.traj files")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--split", choices=("Training", "Validation"), default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max videos (debug)")
    parser.add_argument("--retry-failed", action="store_true", help="Re-download error/missing log entries")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    _ensure_csv(
        SPLITS_URL,
        data_dir / "raw_train_val_splits.csv",
        local_fallback=LOCAL_SPLITS,
    )
    _ensure_csv(METADATA_URL, data_dir / "metadata.csv")

    tasks = _load_splits(data_dir / "raw_train_val_splits.csv", args.split)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print(f"Videos to fetch: {len(tasks)} (split={args.split or 'all'})")
    print(f"Output: {data_dir / 'traj'}")

    ok = missing = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_one, data_dir, fold, vid): (fold, vid)
            for fold, vid in tasks
        }
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            vid, status, detail = fut.result()
            if status == "ok":
                ok += 1
            elif status == "missing":
                missing += 1
            else:
                err += 1
                print(f"ERROR {vid}: {detail}")
            if done % 200 == 0 or done == total:
                print(f"Progress {done}/{total}  ok={ok} missing={missing} error={err}")

    log_path = data_dir / "download_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold", "video_id", "path", "exists"])
        for fold, vid in tasks:
            p = _traj_dest(data_dir, fold, vid)
            w.writerow([fold, vid, str(p), int(p.is_file())])

    print(f"Done. ok={ok} missing={missing} error={err}")
    print(f"Log: {log_path}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
