from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

class SampleFlattener:
    """
    Flatten multi-image samples (attributes as lists) into single-image samples (attributes as single values).
    """
    def __init__(self, args):
        self.args = args or {}
        self.anchor_col = self.args.get("anchor_col", "image")
        default_split_col_list = ['image', 'bboxes_2d', 'obj_tags', 'depth_map', 'pose', "intrinsic"]
        self.split_col_list = self.args.get("split_col_list", default_split_col_list)
        assert self.anchor_col in self.split_col_list, f"anchor_col '{self.anchor_col}' must be in split_col_list {self.split_col_list}"

    @staticmethod
    def _is_sequence_value(value: Any) -> bool:
        """Parquet list columns often load as numpy arrays instead of Python lists."""
        return isinstance(value, (list, tuple, np.ndarray))

    def flatten(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten multi-image samples (attributes as lists) into single-image samples (attributes as single values).
        :param dataset: input dataset (pandas.DataFrame)
        :return: flattened dataset (pandas.DataFrame)
        """
        flat_examples = []

        for idx in tqdm(range(len(dataset)), desc="Flattening samples"):
            sample = dataset.iloc[idx]
            anchor_col = self.anchor_col
            anchor_val = sample.get(anchor_col, None)
            is_multi = self._is_sequence_value(anchor_val)
            if not is_multi:
                flat_examples.append(sample)
                continue

            n_img = len(anchor_val)
            split_col_list = self.split_col_list
            if split_col_list is None:
                raise ValueError("split_col_list must be provided for flattening.")

            # Check that all split_col_list elements are in sample
            missing_cols = [k for k in split_col_list if k not in sample]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} from split_col_list are missing in sample keys: {list(sample.keys())}")

            # Check that all columns to split are lists and have the same length as anchor_column
            for k in split_col_list:
                v = sample.get(k, None)
                if not self._is_sequence_value(v) or len(v) != n_img:
                    raise ValueError(f"Column '{k}' must be a sequence of length {n_img} (same as anchor_column '{anchor_col}'). Got type {type(v)} and length {len(v) if self._is_sequence_value(v) else 'N/A'}.")

            for i in range(n_img):
                new_sample = {}
                for k, v in sample.items():
                    if k in split_col_list:
                        new_sample[k] = v[i]
                    else:
                        new_sample[k] = v
                flat_examples.append(new_sample)

        print(f'>>> Flattened dataset size: {len(flat_examples)} samples')
        return pd.DataFrame(flat_examples)

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Process the dataset to flatten multi-image samples.
        :param dataset: input dataset (pandas.DataFrame)
        :return: flattened dataset (pandas.DataFrame)
        """
        return self.flatten(dataset)