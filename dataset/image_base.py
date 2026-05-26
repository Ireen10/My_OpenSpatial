import io
import math
import os
import re

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from PIL import Image as PILImage

from utils.data_utils import flatten_annotations, strip_empty_structs

HF_REPO_PATTERN = re.compile(r'^[\w-]+/[\w-]+$')

# Parquet path columns that are relative to raw sensor data (e.g. EmbodiedScan data/).
ASSET_PATH_FIELDS = (
    "image",
    "depth_map",
    "intrinsic",
    "pose",
    "axis_align_matrix",
)


def _resolve_asset_path(path, raw_data_root):
    """Join relative asset paths to raw_data_root; leave absolute paths unchanged."""
    if not path or not isinstance(path, str):
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(raw_data_root, path))


def _resolve_asset_value(value, raw_data_root):
    if isinstance(value, str):
        return _resolve_asset_path(value, raw_data_root)
    if isinstance(value, list):
        return [_resolve_asset_value(item, raw_data_root) for item in value]
    return value


class ImageBaseDataset:
    """Base image dataset backed by parquet or HuggingFace Hub."""

    MODALITY = "image"

    def __init__(self, cfg, *, _skip_load=False):
        if not cfg.data_dir:
            raise ValueError("cfg.data_dir is required")
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.raw_data_root = getattr(cfg, "raw_data_root", None)
        if self.raw_data_root:
            self.raw_data_root = os.path.abspath(self.raw_data_root)
        self.data = None if _skip_load else self._load()

    # ------------------------------------------------------------------
    # Load / Override
    # ------------------------------------------------------------------

    def _load(self):
        """Load data from HuggingFace Hub or local parquet."""
        if HF_REPO_PATTERN.match(self.data_dir):
            data = pd.DataFrame(load_dataset(self.data_dir, split="train"))
        else:
            data = pd.read_parquet(self.data_dir, engine="pyarrow", dtype_backend="pyarrow")
        return self._apply_raw_data_root(data)

    def _apply_raw_data_root(self, df):
        """Resolve relative asset paths in path columns against raw_data_root."""
        if not self.raw_data_root or df is None or len(df) == 0:
            return df
        for col in ASSET_PATH_FIELDS:
            if col not in df.columns:
                continue
            df[col] = df[col].apply(
                lambda v: _resolve_asset_value(v, self.raw_data_root)
            )
        return df

    def override_data(self, data_path):
        """Replace in-memory data with another parquet file."""
        try:
            data = pd.read_parquet(data_path, engine="pyarrow", dtype_backend="pyarrow")
            self.data = self._apply_raw_data_root(data)
        except Exception as exc:
            raise ValueError(f"Failed to load parquet: {data_path}") from exc

    # ------------------------------------------------------------------
    # Image format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _bytes_dict_to_pil(img_dict):
        """Convert {"bytes": ...} dict to PIL Image."""
        if isinstance(img_dict, dict) and img_dict.get("bytes"):
            try:
                return PILImage.open(io.BytesIO(img_dict["bytes"]))
            except Exception:
                return img_dict
        return img_dict

    @staticmethod
    def _pil_to_bytes_dict(image):
        """Convert PIL Image to {"bytes": ...} dict."""
        if isinstance(image, PILImage.Image):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return {"bytes": buf.getvalue()}
        return image

    def convert_image_column_to_pil(self, df, col="image"):
        """Convert bytes dicts in a column to PIL objects (in-place)."""
        def _convert(item):
            if item is None:
                return None
            if isinstance(item, dict):
                return self._bytes_dict_to_pil(item)
            if isinstance(item, (list, tuple, np.ndarray)):
                seq = list(item)
                if seq and all(isinstance(x, dict) and "bytes" in x for x in seq):
                    return [self._bytes_dict_to_pil(x) for x in seq]
                return seq
            return item

        df[col] = [_convert(item) for item in df[col]]
        return df

    def pil_convert_to_bytes(self, df):
        """Convert PIL images in all DataFrame columns to bytes dicts."""
        def _is_pil(x):
            return isinstance(x, PILImage.Image) or (
                isinstance(x, list) and all(isinstance(i, PILImage.Image) for i in x))

        for col in df.columns:
            if df[col].apply(_is_pil).any():
                df[col] = df[col].apply(
                    lambda x: [self._pil_to_bytes_dict(i) for i in x]
                    if isinstance(x, list) else self._pil_to_bytes_dict(x))
        return df

    def pil_convert_to_np(self, data):
        """Convert image column from PIL to nested Python lists."""
        images = data["image"]
        if not len(images):
            return data

        if isinstance(images.iloc[0], list):
            data["image"] = [[np.array(img).tolist() for img in row] for row in images]
        else:
            data["image"] = [np.array(img).tolist() for img in images]
        return data

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_data(self, data_path, data=None, annotation_flag=False,
                  batch_size=1000, keep_data_columns=None):
        """Save DataFrame to parquet with optional annotation flattening."""
        if data is None:
            raise ValueError("Data to save is None")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Only pandas DataFrame is supported")

        if annotation_flag:
            keep_data_columns = keep_data_columns or [
                "messages", "QA_images", "question_tags", "question_types"]
            data = flatten_annotations(data, keep_keys=keep_data_columns)
            if len(data) > batch_size:
                data = self._parquet_safe_frame(data)
                self._save_batches(data_path, data, batch_size)
                return
            data = self._parquet_safe_frame(data)
            data.to_parquet(data_path, engine="pyarrow")
            return

        data = self._parquet_safe_frame(data)
        data.to_parquet(data_path, engine="pyarrow")

    @staticmethod
    def _parquet_safe_frame(data: pd.DataFrame) -> pd.DataFrame:
        """Coerce nested metadata / list cells for PyArrow (empty structs, numpy scalars)."""
        if "metadata" not in data.columns:
            return data
        out = data.copy()

        def _clean_meta(m):
            if m is None or (isinstance(m, float) and pd.isna(m)):
                return m
            if isinstance(m, list) and m and isinstance(m[0], dict):
                return [strip_empty_structs(x) for x in m]
            if isinstance(m, dict):
                return strip_empty_structs(m)
            return m

        out["metadata"] = out["metadata"].apply(_clean_meta)
        return out

    @staticmethod
    def _save_batches(data_path, data, batch_size):
        """Save DataFrame into multiple parquet parts."""
        base = os.path.splitext(data_path)[0]
        for i in range(math.ceil(len(data) / batch_size)):
            batch = data.iloc[i * batch_size:(i + 1) * batch_size]
            batch.to_parquet(f"{base}_part_{i}.parquet", engine="pyarrow")

    def convert_to_hf_dataset(self, data):
        """Convert pandas DataFrame to HuggingFace Dataset."""
        return Dataset.from_dict(data.to_dict(orient="list"))
