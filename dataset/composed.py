"""Compose separate dataset backends for load vs save."""

from __future__ import annotations


class ComposedDataset:
    """
    Pipeline facade: input backend loads; output backend writes.

    Typical M7 pattern (aggregate parquet → upstream bundle):

        dataset:
          input_dataset_name: image_base
          output_dataset_name: jsonl_base
          export_bundle: true
          export_dir: export

    When backends differ:
      - input writes ``data.parquet`` (stage chaining / task summary)
      - output writes upstream JSONL+tar under ``export_dir`` when ``export_bundle`` is set
      - if output has ``export_bundle``, input still writes parquet unless
        ``skip_input_parquet_on_export: true``
    """

    def __init__(self, input_backend, output_backend):
        self._input = input_backend
        self._output = output_backend

    @property
    def data(self):
        return self._input.data

    @data.setter
    def data(self, value):
        self._input.data = value

    @property
    def cfg(self):
        return self._input.cfg

    def override_data(self, data_path):
        return self._input.override_data(data_path)

    def save_data(self, data_path, data=None, **kwargs):
        if self._output is self._input:
            return self._output.save_data(data_path, data, **kwargs)

        skip_input = bool(
            getattr(self._output, "export_bundle", False)
            and getattr(self._output.cfg, "skip_input_parquet_on_export", False)
        )
        if not skip_input:
            self._input.save_data(data_path, data, **kwargs)
        self._output.data = data
        return self._output.save_data(data_path, data, **kwargs)
