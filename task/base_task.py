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
        from concurrent.futures import ThreadPoolExecutor

        num_workers = self.args.get('num_workers', 8)
        examples = list(enumerate(dataset.to_dict('records')))
        print(f"  [{type(self).__name__}] {len(examples)} examples, {num_workers} workers",
              flush=True)

        processed = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(
                lambda item: self.apply_transform(item[1], item[0]),
                examples,
            )
            for result, flag in tqdm.tqdm(results, total=len(examples),
                                          desc="Processing examples"):
                if flag:
                    processed.append(result)

        print(f"  [{type(self).__name__}] {len(processed)}/{len(examples)} passed", flush=True)
        return pd.DataFrame(processed).reset_index(drop=True)
