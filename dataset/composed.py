"""Compose separate dataset backends for load vs save."""


class ComposedDataset:
    """
    Pipeline facade: input backend owns in-memory data and loads; output backend writes.

    When backends differ, save_data writes Parquet at ``data_path`` (for stage chaining)
    and also runs the output backend (e.g. JSONL alongside ``data.jsonl``).
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

        self._input.save_data(data_path, data, **kwargs)
        self._output.data = data
        return self._output.save_data(data_path, data, **kwargs)
