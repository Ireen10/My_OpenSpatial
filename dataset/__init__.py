from dataset.composed import ComposedDataset
from dataset.image_base import ImageBaseDataset
from dataset.jsonl_base import JsonlBaseDataset

IMAGE_DATASETS = {
    "image_base": ImageBaseDataset,
    "jsonl_base": JsonlBaseDataset,
}

VIDEO_DATASETS = {}


def _resolve_dataset_name(cfg, role):
    """Resolve input/output role: role-specific key > legacy dataset_name."""
    role_key = f"{role}_dataset_name"
    name = getattr(cfg, role_key, None) or getattr(cfg, "dataset_name", None)
    if not name:
        raise ValueError(
            f"Dataset config must set '{role_key}' or 'dataset_name' (legacy shorthand for both)."
        )
    return name


def _get_dataset_class(cfg, dataset_name):
    registry = IMAGE_DATASETS if cfg.modality == "image" else VIDEO_DATASETS
    cls = registry.get(dataset_name)
    if cls is None:
        raise ValueError(
            f"Dataset [{dataset_name}] not found under [{cfg.modality}] modality"
        )
    return cls


def _instantiate_dataset(cfg, dataset_name, *, load_data):
    cls = _get_dataset_class(cfg, dataset_name)
    return cls(cfg, _skip_load=not load_data)


def build_dataset(cfg, dataset_name=None):
    """
    Build pipeline dataset with optional decoupled input/output backends.

    YAML (backward compatible):
        dataset_name: image_base

    YAML (decoupled):
        input_dataset_name: image_base
        output_dataset_name: jsonl_base
    """
    if dataset_name and not getattr(cfg, "dataset_name", None):
        cfg.dataset_name = dataset_name
    input_name = _resolve_dataset_name(cfg, "input")
    output_name = _resolve_dataset_name(cfg, "output")

    input_ds = _instantiate_dataset(cfg, input_name, load_data=True)
    if input_name == output_name:
        return input_ds

    output_ds = _instantiate_dataset(cfg, output_name, load_data=False)
    return ComposedDataset(input_ds, output_ds)
