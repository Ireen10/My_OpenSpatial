"""Base class for all OpenSpatial task stages."""

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
        for idx in tqdm.tqdm(range(len(dataset)), total=len(dataset),
                             desc="Processing examples"):
            example = dataset.iloc[idx].to_dict()
            result, flag = self.apply_transform(example, idx)
            if flag:
                processed.append(result)

        return pd.DataFrame(processed).reset_index(drop=True)

    def _run_multi_processing(self, dataset):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_workers = self.args.get('num_workers', 8)
        n = len(dataset)
        print(f"  [{type(self).__name__}] {n} examples, {num_workers} workers",
              flush=True)

        def _work(idx):
            example = dataset.iloc[idx].to_dict()
            return self.apply_transform(example, idx)

        processed = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_work, idx) for idx in range(n)]
            for fut in tqdm.tqdm(as_completed(futures), total=n,
                                 desc="Processing examples"):
                result, flag = fut.result()
                if flag:
                    processed.append(result)

        print(f"  [{type(self).__name__}] {len(processed)}/{n} passed", flush=True)
        return pd.DataFrame(processed).reset_index(drop=True)
