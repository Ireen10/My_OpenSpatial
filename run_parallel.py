"""run_parallel.py – Parallel shard runner for OpenSpatial preprocessing.

Splits the input dataset across N independent pipeline subprocesses, each
pinned to a dedicated NPU device.  Because every subprocess handles only 1/N
of the data, all stages (both CPU-bound and NPU-bound) achieve ~N× throughput.

All hardware parameters (devices, replicas_per_device, cpu_workers) are read
directly from the YAML config.  Only --config, --output_dir, and
--num_pipelines are required; the remaining flags are pure overrides.

Minimal invocation:

    python run_parallel.py \\
        --config config/preprocessing/demo_preprocessing_embodiedscan_sam3.yaml \\
        --output_dir /data/output \\
        --num_pipelines 2

Override any YAML value when needed, e.g. to test a different replica count
without editing the file:

    python run_parallel.py \\
        --config config/preprocessing/demo_preprocessing_embodiedscan_sam3.yaml \\
        --output_dir /data/output \\
        --num_pipelines 2 \\
        --replicas_per_device 3

Output layout:
    <output_dir>/worker_0/    ← shard 0 results
    <output_dir>/worker_1/    ← shard 1 results
    ...

Each worker_N directory has the same sub-structure as a normal run.py output.
You can load and concatenate the final data.parquet files from all workers for
downstream use.
"""

import argparse
import copy
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dump_yaml(cfg: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)


def _iter_all_tasks(stages):
    """Yield every task dict found in stages (handles dict or list-of-dicts)."""
    if isinstance(stages, dict):
        for tasks in stages.values():
            if isinstance(tasks, list):
                yield from (t for t in tasks if isinstance(t, dict))
    elif isinstance(stages, list):
        for entry in stages:
            if isinstance(entry, dict):
                yield from _iter_all_tasks(entry)


def _infer_devices(config_dict: dict) -> list[str]:
    """Read the 'device' field from the first NPU-bound task in the YAML."""
    stages = config_dict.get("pipeline", {}).get("stages", {})
    for task in _iter_all_tasks(stages):
        if "device" in task:
            return [d.strip() for d in str(task["device"]).split(",")]
    return ["cpu"]


def _infer_replicas_per_device(config_dict: dict) -> int:
    """Read 'replicas_per_device' from the first NPU-bound task in the YAML."""
    stages = config_dict.get("pipeline", {}).get("stages", {})
    for task in _iter_all_tasks(stages):
        if "replicas_per_device" in task:
            return int(task["replicas_per_device"])
    return 1


def _infer_cpu_workers(config_dict: dict) -> int:
    """Read 'num_workers' from the first CPU-bound task that declares it."""
    stages = config_dict.get("pipeline", {}).get("stages", {})
    for task in _iter_all_tasks(stages):
        if "device" not in task and "num_workers" in task:
            return int(task["num_workers"])
    return 8


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def _discover_parquets(directory: str) -> list[str]:
    """Return all .parquet files under a directory, sorted for reproducibility."""
    import glob as _glob
    # Prefer top-level files first; fall back to recursive search
    files = sorted(_glob.glob(os.path.join(directory, "*.parquet")))
    if not files:
        files = sorted(_glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True))
    return files


def _split_list(paths: list, n: int) -> list[list]:
    """Distribute a list of paths across N workers (round-robin, no data copying)."""
    chunks: list[list] = [[] for _ in range(n)]
    for i, p in enumerate(paths):
        chunks[i % n].append(p)
    return [c for c in chunks if c]


def _split_single_parquet(data_path: str, n: int, tmpdir: str) -> list[str]:
    """Row-split a single parquet into N shards.  Only used when data_dir is one file.

    Preserves the original pyarrow schema by reading and writing with the same
    engine/backend settings used by ImageBaseDataset.
    """
    import pyarrow.parquet as pq
    table = pq.read_table(data_path)
    total = len(table)
    chunk = (total + n - 1) // n  # ceiling division

    shard_paths = []
    for i in range(n):
        shard = table.slice(i * chunk, min(chunk, total - i * chunk))
        if len(shard) == 0:
            continue
        out = os.path.join(tmpdir, f"shard_{i:03d}.parquet")
        pq.write_table(shard, out)
        shard_paths.append(out)
        print(f"  Shard {i}: {len(shard):,} rows → {out}")
    return shard_paths


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------

def _patch_stages(stages: dict | list, localization_device: str,
                  replicas_per_device: int, cpu_workers: int) -> None:
    """Recursively patch task configs for this worker's device assignment.

    Rules:
    - localization_stage tasks with a 'device' key → pin to localization_device,
      set replicas_per_device and num_workers accordingly.
    - All other tasks → ensure use_multi_processing=true and num_workers=cpu_workers.
    """
    if isinstance(stages, list):
        # list of {stage_name: [...]} dicts (DuplicateKeySafe format)
        for entry in stages:
            if isinstance(entry, dict):
                _patch_stages(entry, localization_device, replicas_per_device, cpu_workers)
        return

    if not isinstance(stages, dict):
        return

    for stage_name, tasks in stages.items():
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if "device" in task:
                # NPU-bound stage (localization)
                task["device"] = localization_device
                task["replicas_per_device"] = replicas_per_device
                task["num_workers"] = replicas_per_device
                task["use_multi_processing"] = True
            else:
                # CPU-bound stage (filter, scene_fusion, group, …)
                task.setdefault("use_multi_processing", True)
                task.setdefault("num_workers", cpu_workers)


def _make_worker_config(base_cfg: dict, worker_idx: int, data_dir,
                        device: str, replicas_per_device: int,
                        cpu_workers: int) -> dict:
    """Return a deep-copied config dict patched for worker_idx."""
    cfg = copy.deepcopy(base_cfg)

    # Patch dataset
    cfg["dataset"]["data_dir"] = data_dir

    # Patch pipeline stages
    stages = cfg.get("pipeline", {}).get("stages", {})
    _patch_stages(stages, device, replicas_per_device, cpu_workers)

    return cfg


# ---------------------------------------------------------------------------
# Subprocess launcher
# ---------------------------------------------------------------------------

def _launch_worker(worker_idx: int, config_path: str, output_dir: str,
                   run_script: str, log_path: str):
    """Launch a worker subprocess with stdout/stderr piped for live streaming.

    Returns (proc, log_fh) — the caller is responsible for starting a
    _stream_worker thread and closing log_fh after the thread finishes.
    """
    cmd = [
        sys.executable, run_script,
        "--config", config_path,
        "--output_dir", output_dir,
    ]
    log_fh = open(log_path, "w", encoding="utf-8")
    print(f"  [worker {worker_idx}] cmd: {' '.join(cmd)}")
    print(f"  [worker {worker_idx}] log: {log_path}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc, log_fh


def _stream_worker(worker_idx: int, proc: subprocess.Popen, log_fh) -> None:
    """Forward worker output to both the log file and the parent terminal.

    Each line is prefixed with ``[W<worker_idx>]`` so output from multiple
    workers is distinguishable without being garbled.  tqdm in non-TTY (pipe)
    mode writes one ``\\n``-terminated line per update, so progress bars appear
    as scrolling lines rather than in-place rewrites.  All ``>>> Running Task``
    stage headers and summary lines are forwarded verbatim.
    """
    prefix = f"[W{worker_idx}]"
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace")
        log_fh.write(line)
        log_fh.flush()
        sys.stdout.write(f"{prefix} {line}")
        sys.stdout.flush()
    proc.wait()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel shard runner for OpenSpatial preprocessing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True,
                        help="Path to the pipeline YAML config (same as run.py).")
    parser.add_argument("--output_dir", required=True,
                        help="Root output directory; worker results go under worker_N/.")
    parser.add_argument("--num_pipelines", type=int, default=2,
                        help="Number of parallel pipeline processes (default: 2).")
    parser.add_argument("--devices", type=str, default=None,
                        help="Space-separated NPU device list, e.g. 'npu:0 npu:1'. "
                             "Default: read from the 'device' field in the YAML.")
    parser.add_argument("--replicas_per_device", type=int, default=None,
                        help="SAM3 replicas per NPU device per worker. "
                             "Default: read from 'replicas_per_device' in the YAML.")
    parser.add_argument("--cpu_workers", type=int, default=None,
                        help="num_workers for CPU-bound stages per pipeline. "
                             "Default: read from the first CPU stage 'num_workers' in the YAML. "
                             "Recommended value ≈ vCPUs / num_pipelines.")
    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    config_path = os.path.abspath(args.config)
    output_root = os.path.abspath(args.output_dir)
    run_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")

    if not os.path.isfile(run_script):
        sys.exit(f"[ERROR] Cannot find run.py at {run_script}")

    os.makedirs(output_root, exist_ok=True)

    # ── Load base config ─────────────────────────────────────────────────────
    base_cfg = _load_yaml(config_path)
    n = args.num_pipelines

    # ── Resolve hardware params: CLI arg > YAML value > built-in fallback ────
    devices = args.devices.split() if args.devices else _infer_devices(base_cfg)
    replicas_per_device = args.replicas_per_device or _infer_replicas_per_device(base_cfg)
    cpu_workers = args.cpu_workers or _infer_cpu_workers(base_cfg)

    print(f"[run_parallel] num_pipelines={n}  devices={devices}  "
          f"replicas_per_device={replicas_per_device}  "
          f"cpu_workers={cpu_workers}  "
          f"(all read from YAML unless overridden via CLI)")

    # ── Split data ───────────────────────────────────────────────────────────
    data_dir = base_cfg["dataset"]["data_dir"]
    tmpdir = tempfile.mkdtemp(prefix="openspatial_shards_")
    needs_cleanup = False  # only True when we wrote temp parquet shards

    if os.path.isdir(data_dir):
        # ── Directory with multiple parquet files: distribute files directly ──
        all_files = _discover_parquets(data_dir)
        if not all_files:
            sys.exit(f"[ERROR] No .parquet files found under {data_dir}")
        print(f"[run_parallel] Found {len(all_files)} parquet file(s) in {data_dir} "
              f"→ distributing across {n} workers (no row-splitting)")
        raw_chunks = _split_list(all_files, n)
        data_shards = [c[0] if len(c) == 1 else c for c in raw_chunks]

    elif isinstance(data_dir, list):
        # ── Explicit list of parquet files: distribute files directly ─────────
        print(f"[run_parallel] Distributing {len(data_dir)} listed file(s) across "
              f"{n} workers (no row-splitting)")
        raw_chunks = _split_list(data_dir, n)
        data_shards = [c[0] if len(c) == 1 else c for c in raw_chunks]

    else:
        # ── Single parquet file: must row-split into temp shards ──────────────
        print(f"[run_parallel] Single parquet file detected — row-splitting into "
              f"{n} shards (tmpdir: {tmpdir}) …")
        print("  TIP: place your parquet files in a directory and set "
              "data_dir to that path to skip this step.")
        shard_paths = _split_single_parquet(data_dir, n, tmpdir)
        data_shards = shard_paths
        needs_cleanup = True

    actual_n = len(data_shards)
    if actual_n < n:
        print(f"[run_parallel] WARNING: only {actual_n} non-empty shards for {n} pipelines; "
              f"running {actual_n} workers.")
        n = actual_n

    # ── Write per-worker configs ─────────────────────────────────────────────
    worker_cfgs = []
    for i in range(n):
        device = devices[i % len(devices)]
        cfg = _make_worker_config(
            base_cfg,
            worker_idx=i,
            data_dir=data_shards[i],
            device=device,
            replicas_per_device=replicas_per_device,
            cpu_workers=cpu_workers,
        )
        cfg_path = os.path.join(tmpdir, f"config_worker_{i:03d}.yaml")
        _dump_yaml(cfg, cfg_path)
        worker_output = os.path.join(output_root, f"worker_{i}")
        os.makedirs(worker_output, exist_ok=True)
        log_path = os.path.join(worker_output, "run.log")
        worker_cfgs.append((i, cfg_path, worker_output, log_path))
        print(f"  Worker {i}: device={device}, "
              f"data_dir={data_shards[i] if isinstance(data_shards[i], str) else f'[{len(data_shards[i])} files]'}, "
              f"output={worker_output}")

    # ── Launch all workers; one streaming thread per worker ──────────────────
    print(f"\n[run_parallel] Launching {n} pipeline subprocesses …")
    print(f"[run_parallel] Output prefixed [W0], [W1], … per worker; "
          f"logs also saved to {output_root}/worker_*/run.log\n")
    procs = []
    threads = []
    log_fhs = []
    for i, cfg_path, worker_output, log_path in worker_cfgs:
        proc, log_fh = _launch_worker(i, cfg_path, worker_output, run_script, log_path)
        t = threading.Thread(
            target=_stream_worker, args=(i, proc, log_fh), daemon=True, name=f"stream-w{i}"
        )
        t.start()
        procs.append((i, proc, log_path))
        threads.append(t)
        log_fhs.append(log_fh)

    # ── Wait for all streaming threads (they exit when their proc exits) ─────
    for t in threads:
        t.join()

    # Close log file handles
    for fh in log_fhs:
        fh.close()

    exit_codes = {i: proc.poll() for i, proc, _ in procs}
    for i, rc in exit_codes.items():
        log_path = next(lp for wi, _, lp in procs if wi == i)
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  [worker {i}] {status}  →  {log_path}")

    # ── Cleanup temp dir (worker configs + optional row-split shards) ────────
    shutil.rmtree(tmpdir, ignore_errors=True)

    failures = [i for i, rc in exit_codes.items() if rc != 0]
    if failures:
        print(f"\n[run_parallel] {len(failures)} worker(s) failed: {failures}")
        sys.exit(1)

    print(f"\n[run_parallel] All {n} workers finished successfully.")
    print(f"Results are in: {output_root}/worker_*/")
    print("To load the combined final parquet (example):")
    print(f"  import pandas as pd, glob")
    print(f"  dfs = [pd.read_parquet(p) for p in sorted(glob.glob('{output_root}/worker_*/*/data.parquet'))]")
    print(f"  combined = pd.concat(dfs, ignore_index=True)")


if __name__ == "__main__":
    main()
