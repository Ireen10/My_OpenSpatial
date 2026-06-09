"""Base class for all OpenSpatial task stages."""

import time

import tqdm
import pandas as pd


class BaseTask:
    """
    Root base class for all tasks.

    Provides:
        - run(dataset) — standard DataFrame iteration + optional multi-threading
        - _run_multi_processing(dataset) — ThreadPoolExecutor parallel execution

    Subclasses must override:
        - apply_transform(self, example, idx) -> (example, bool)
    """

    def __init__(self, args):
        self.args = args
        self.use_multi_processing = args.get("use_multi_processing", False)
        num_workers = args.get("num_workers", 8)
        if num_workers > 1 and not self.use_multi_processing:
            print(
                f"  WARNING: num_workers={num_workers} ignored — "
                f"use_multi_processing is false; running single-threaded.",
                flush=True,
            )

    def apply_transform(self, example, idx):
        raise NotImplementedError

    def run(self, dataset):
        if self.use_multi_processing:
            return self._run_multi_processing(dataset)

        processed = []
        errors = 0
        for idx in tqdm.tqdm(range(len(dataset)), total=len(dataset),
                             desc="Processing examples"):
            example = dataset.iloc[idx].to_dict()
            try:
                result, flag = self.apply_transform(example, idx)
            except Exception as exc:
                errors += 1
                print(f"  [error] example {idx}: {exc}", flush=True)
                continue
            if flag:
                processed.append(result)

        if errors:
            print(f"  [{type(self).__name__}] {errors} example(s) failed", flush=True)
        return pd.DataFrame(processed).reset_index(drop=True)

    def _run_multi_processing(self, dataset):
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        num_workers = self.args.get('num_workers', 8)
        slow_log_s = float(self.args.get("slow_example_log_s", 120))
        n = len(dataset)
        window = max(num_workers * 2, num_workers + 1)
        print(
            f"  [{type(self).__name__}] {n} examples, {num_workers} workers "
            f"(window={window})",
            flush=True,
        )

        def _work(idx):
            try:
                from task.annotation.core.thread_rng import seed_thread_rng
                seed_thread_rng(idx)
            except ImportError:
                pass
            t0 = time.perf_counter()
            example = dataset.iloc[idx].to_dict()
            result, flag = self.apply_transform(example, idx)
            elapsed = time.perf_counter() - t0
            if elapsed >= slow_log_s:
                print(
                    f"  [slow] example {idx} took {elapsed:.1f}s",
                    flush=True,
                )
            return idx, result, flag

        processed_by_idx = {}
        pending = set()
        next_idx = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pbar = tqdm.tqdm(total=n, desc="Processing examples")
            while next_idx < n or pending:
                while next_idx < n and len(pending) < window:
                    pending.add(executor.submit(_work, next_idx))
                    next_idx += 1
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        idx, result, flag = fut.result()
                    except Exception as exc:
                        errors += 1
                        print(f"  [error] worker failed: {exc}", flush=True)
                        pbar.update(1)
                        continue
                    if flag:
                        processed_by_idx[idx] = result
                    pbar.update(1)
            pbar.close()

        processed = [processed_by_idx[i] for i in sorted(processed_by_idx)]
        print(
            f"  [{type(self).__name__}] {len(processed)}/{n} passed"
            + (f", {errors} failed" if errors else ""),
            flush=True,
        )
        return pd.DataFrame(processed).reset_index(drop=True)
