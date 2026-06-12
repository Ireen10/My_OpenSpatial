import math
import os
import queue
import threading
import warnings

# Suppress SyntaxWarnings from Ascend CANN internal libraries.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"tbe\..*")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from task.base_task import BaseTask


# ---------------------------------------------------------------------------
# NPU helpers
# ---------------------------------------------------------------------------

def _try_import_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        return False


def _npu_available() -> bool:
    if not _try_import_torch_npu():
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


def _is_npu_device(device: str) -> bool:
    return str(device).startswith("npu")


_NPU_CONFIGURED = False


def _configure_npu_once() -> None:
    """Apply Ascend NPU runtime options (idempotent)."""
    global _NPU_CONFIGURED
    if _NPU_CONFIGURED or not _npu_available():
        return
    _NPU_CONFIGURED = True
    import torch_npu  # noqa: F401

    if hasattr(torch.npu, "set_compile_mode"):
        torch.npu.set_compile_mode(jit_compile=False)
    opt = {
        "ACL_PRECISION_MODE": "must_keep_origin_dtype",
        "ACL_OP_SELECT_IMPL_MODE": "high_precision",
    }
    torch_npu.npu.set_option(opt)
    if hasattr(torch.npu, "config") and hasattr(torch.npu.config, "allow_internal_format"):
        torch.npu.config.allow_internal_format = False
    if hasattr(torch.npu, "conv") and hasattr(torch.npu.conv, "allow_hf32"):
        torch.npu.conv.allow_hf32 = False
    if hasattr(torch.npu, "matmul") and hasattr(torch.npu.matmul, "allow_hf32"):
        torch.npu.matmul.allow_hf32 = False


def _try_patch_roi_align_for_npu() -> bool:
    """Replace torchvision.ops.roi_align with torch_npu.npu_roi_align on NPU."""
    if not _npu_available():
        return False
    try:
        import torchvision.ops as _tvops
        import torch_npu  # noqa: F401
        _npu_fn = torch_npu.npu_roi_align
    except (ImportError, AttributeError):
        return False

    if getattr(_tvops.roi_align, "_npu_patched", False):
        return True

    def _roi_align_npu(
        input, boxes, output_size,
        spatial_scale=1.0, sampling_ratio=-1, aligned=False,
    ):
        if isinstance(output_size, (int, float)):
            pooled_h = pooled_w = int(output_size)
        else:
            pooled_h, pooled_w = int(output_size[0]), int(output_size[1])
        roi_end_mode = 1 if aligned else 0
        npu_sample_num = max(0, sampling_ratio)

        if isinstance(boxes, (list, tuple)):
            per_image_boxes = boxes
        else:
            if boxes.shape[0] == 0:
                return input.new_zeros((0, input.shape[1], pooled_h, pooled_w))
            per_image_boxes = [
                boxes[boxes[:, 0] == i, 1:] for i in range(input.shape[0])
            ]

        results = []
        for i, b in enumerate(per_image_boxes):
            if b.shape[0] == 0:
                continue
            rois_i = torch.cat([b.new_zeros((b.shape[0], 1)), b.float()], dim=1)
            results.append(_npu_fn(
                input[i : i + 1], rois_i,
                spatial_scale, pooled_h, pooled_w, npu_sample_num, roi_end_mode,
            ))

        if not results:
            return input.new_zeros((0, input.shape[1], pooled_h, pooled_w))
        return torch.cat(results, dim=0)

    _roi_align_npu._npu_patched = True
    _tvops.roi_align = _roi_align_npu
    try:
        import torchvision.ops.poolers as _poolers
        if hasattr(_poolers, "roi_align"):
            _poolers.roi_align = _roi_align_npu
    except (ImportError, AttributeError):
        pass
    return True


def _parse_devices(device_raw) -> list[str]:
    if isinstance(device_raw, (list, tuple)):
        return [str(d).strip() for d in device_raw if str(d).strip()]
    return [d.strip() for d in str(device_raw).split(",") if d.strip()]


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if _npu_available():
        return "npu:0"
    return "cpu"


def _set_npu_device(device: str) -> None:
    if not _is_npu_device(device):
        return
    if not _try_import_torch_npu():
        raise ImportError("torch_npu is required for device='npu:*'.")
    dev_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    torch.npu.set_device(dev_id)


# ---------------------------------------------------------------------------
# NPU-safe SAM3 vision cross-attention (explicit matmul + fp32 softmax)
# ---------------------------------------------------------------------------

class SafeSam3Attention(nn.Module):
    """Drop-in replacement for Sam3Attention that avoids fused NPU attention ops."""

    def __init__(self, attn: nn.Module, num_heads: int | None = None, head_dim: int | None = None):
        super().__init__()
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
        if self.o_proj is None:
            raise AttributeError("Sam3Attention has no o_proj/out_proj.")

        self.embed_dim = self.q_proj.out_features

        if num_heads is None:
            num_heads = (
                getattr(attn, "num_heads", None)
                or getattr(attn, "num_attention_heads", None)
                or getattr(attn, "n_heads", None)
            )
        if head_dim is None:
            head_dim = getattr(attn, "head_dim", None) or getattr(attn, "attention_head_size", None)

        if head_dim is None:
            scaling = getattr(attn, "scaling", None) or getattr(attn, "scale", None)
            if scaling is not None:
                try:
                    head_dim = int(round(1.0 / (float(scaling) ** 2)))
                except Exception:
                    head_dim = None

        if num_heads is None and head_dim is not None:
            num_heads = self.embed_dim // head_dim

        if num_heads is None:
            for h in (8, 16, 4, 32):
                if self.embed_dim % h == 0:
                    num_heads = h
                    break

        if head_dim is None:
            head_dim = self.embed_dim // int(num_heads)

        if self.embed_dim != int(num_heads) * int(head_dim):
            raise ValueError(
                f"Bad heads: embed_dim={self.embed_dim}, num_heads={num_heads}, head_dim={head_dim}"
            )

        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _pick_hidden_states(self, args, kwargs):
        for key in ("hidden_states", "query", "x"):
            if key in kwargs and torch.is_tensor(kwargs[key]):
                return kwargs[key]
        for a in args:
            if torch.is_tensor(a) and a.ndim == 3 and a.shape[-1] == self.embed_dim:
                return a
        for v in kwargs.values():
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[-1] == self.embed_dim:
                return v
        return None

    def _pick_kv_states(self, hidden_states, args, kwargs):
        for key in ("key_value_states", "encoder_hidden_states", "memory", "context"):
            if key in kwargs and torch.is_tensor(kwargs[key]) and kwargs[key].ndim == 3:
                return kwargs[key]

        cand = []
        lq = hidden_states.shape[1]
        for a in args:
            if torch.is_tensor(a) and a.ndim == 3 and a.shape[-1] == self.embed_dim:
                cand.append((a.shape[1], a))
        for v in kwargs.values():
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[-1] == self.embed_dim:
                cand.append((v.shape[1], v))

        cand = [(l, t) for l, t in cand if t is not hidden_states]
        diff = [(l, t) for l, t in cand if l != lq]
        if diff:
            return max(diff, key=lambda x: x[0])[1]
        return hidden_states

    def _pick_attention_mask(self, batch_size, lq, lk, args, kwargs):
        if "attention_mask" in kwargs and torch.is_tensor(kwargs["attention_mask"]):
            return kwargs["attention_mask"]

        for a in args:
            if torch.is_tensor(a) and a.ndim in (2, 3, 4):
                t = a
                if t.ndim == 2 and t.shape == (batch_size, lk):
                    return t
                if t.ndim == 4 and t.shape[-1] == lk:
                    return t
                if t.ndim == 3 and t.shape[0] == batch_size and t.shape[-1] == lk:
                    return t
        for v in kwargs.values():
            if torch.is_tensor(v) and v.ndim in (2, 3, 4):
                t = v
                if t.ndim == 2 and t.shape == (batch_size, lk):
                    return t
                if t.ndim == 4 and t.shape[-1] == lk:
                    return t
                if t.ndim == 3 and t.shape[0] == batch_size and t.shape[-1] == lk:
                    return t
        return None

    def _apply_mask(self, scores, mask):
        if mask is None:
            return scores
        batch_size, _, lq, lk = scores.shape
        m = mask
        if m.dtype == torch.bool:
            try:
                return scores.masked_fill(m, float("-inf"))
            except Exception:
                if m.ndim == 2:
                    return scores.masked_fill(m[:, None, None, :], float("-inf"))
                return scores

        if m.ndim == 2 and m.shape == (batch_size, lk):
            mm = m
            if mm.dtype in (torch.int32, torch.int64, torch.uint8) or (mm.min() >= 0 and mm.max() <= 1.0):
                mm = (1.0 - mm.float()) * (-1e4)
            else:
                mm = mm.float()
            m4 = mm[:, None, None, :]
        elif m.ndim == 4:
            m4 = m.float()
        elif m.ndim == 3:
            if m.shape[1] == 1 and m.shape[2] == lk:
                m4 = m[:, None, :, :].float()
            elif m.shape[1] == lq and m.shape[2] == lk:
                m4 = m[:, None, :, :].float()
            else:
                return scores
        else:
            return scores

        try:
            return scores + m4.to(scores.device, dtype=scores.dtype)
        except Exception:
            return scores

    def forward(self, *args, **kwargs):
        hidden_states = self._pick_hidden_states(args, kwargs)
        if hidden_states is None:
            raise TypeError("SafeSam3Attention: cannot find hidden_states/query in args/kwargs")

        key_value_states = self._pick_kv_states(hidden_states, args, kwargs)

        attn_bias = None
        for key in ("attn_bias", "attention_bias", "position_bias", "rpb", "bias"):
            if key in kwargs and torch.is_tensor(kwargs[key]) and kwargs[key].ndim == 4:
                attn_bias = kwargs[key]
                break

        q = self.q_proj(hidden_states)
        k = self.k_proj(key_value_states)
        v = self.v_proj(key_value_states)

        batch_size, lq, embed_dim = q.shape
        lk = k.shape[1]
        num_heads, head_dim = self.num_heads, self.head_dim

        q = q.view(batch_size, lq, num_heads, head_dim).transpose(1, 2).contiguous()
        k = k.view(batch_size, lk, num_heads, head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, lk, num_heads, head_dim).transpose(1, 2).contiguous()

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            scores = scores + attn_bias.to(scores.device, dtype=scores.dtype)

        attention_mask = self._pick_attention_mask(batch_size, lq, lk, args, kwargs)
        scores = self._apply_mask(scores, attention_mask)

        s = scores.float()
        s = s - s.max(dim=-1, keepdim=True).values
        probs = torch.softmax(s, dim=-1).to(dtype=v.dtype)

        ctx = torch.matmul(probs, v)
        ctx = ctx.transpose(1, 2).contiguous().view(batch_size, lq, embed_dim)
        return (self.o_proj(ctx), None)


def _replace_vision_cross_attn(model, upto_layers: int = 6) -> None:
    for i in range(upto_layers):
        model.detr_decoder.layers[i].vision_cross_attn = SafeSam3Attention(
            model.detr_decoder.layers[i].vision_cross_attn,
        )


# ---------------------------------------------------------------------------
# Model loading & post-processing
# ---------------------------------------------------------------------------

def _load_sam3_replica(segmenter_model: str, load_kwargs: dict, device: str):
    """Load Sam3Processor + Sam3Model. Requires transformers>=5.0.0."""
    if _is_npu_device(device):
        _configure_npu_once()
        _try_patch_roi_align_for_npu()

    is_local = (
        os.path.isabs(segmenter_model)
        or segmenter_model.startswith("./")
        or segmenter_model.startswith("../")
    )
    if is_local and not os.path.isdir(segmenter_model):
        raise FileNotFoundError(
            f"SAM3 weights directory not found: {segmenter_model!r}\n"
            "Check 'segmenter_model' in your YAML — absolute path or Hub id 'facebook/sam3'."
        )

    try:
        from transformers import Sam3Processor, Sam3Model
    except ImportError as e:
        raise ImportError(
            "Sam3Processor/Sam3Model not found. Install transformers>=5.0.0."
        ) from e

    if _is_npu_device(device) and "torch_dtype" not in load_kwargs:
        load_kwargs = {**load_kwargs, "torch_dtype": torch.float32}

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are using a model of type sam3_video",
            category=UserWarning,
        )
        try:
            proc = Sam3Processor.from_pretrained(segmenter_model)
            model = Sam3Model.from_pretrained(segmenter_model, **load_kwargs).to(device)
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"Failed to load Sam3Model from {segmenter_model!r}."
            ) from e

    if _is_npu_device(device):
        _replace_vision_cross_attn(model)

    model.eval()
    return proc, model


def _target_sizes(inputs) -> list:
    sizes = inputs["original_sizes"]
    return sizes.tolist() if hasattr(sizes, "tolist") else list(sizes)


def _post_process(processor, outputs, score_threshold: float, target_sizes: list):
    return processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=0.5,
        target_sizes=target_sizes,
    )


def _load_coarse_mask(path: str) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127


def _tags_per_box(tags, n_boxes: int) -> list[str]:
    if n_boxes <= 0:
        return []
    if tags is None:
        return ["object"] * n_boxes
    if isinstance(tags, np.ndarray):
        tags = tags.tolist()
    elif not isinstance(tags, (list, tuple)):
        tags = [tags]
    out: list[str] = []
    for i in range(n_boxes):
        tag = tags[i] if i < len(tags) else None
        if tag is not None and str(tag).strip():
            out.append(str(tag).strip())
        else:
            out.append("object")
    return out


def _unique_tags_preserve_order(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _mask_to_numpy(mask) -> np.ndarray:
    if hasattr(mask, "float"):
        mask = mask.float()
    arr = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float32)


def _mask_box_iou(mask: np.ndarray, box: np.ndarray) -> float:
    """IoU between a binary mask and an axis-aligned box region."""
    h, w = mask.shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 < x1 or y2 < y1:
        return 0.0
    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    mask_bin = mask > 0
    mask_area = float(mask_bin.sum())
    inter = float(mask_bin[y1 : y2 + 1, x1 : x2 + 1].sum())
    union = mask_area + box_area - inter
    return inter / union if union > 0 else 0.0


def _extract_seg_masks_scores(seg_item) -> tuple[list[np.ndarray], list[float]]:
    scores, masks = seg_item.get("scores"), seg_item.get("masks")
    if scores is None or masks is None or len(scores) == 0:
        return [], []
    scores_np = (
        scores.float().cpu().numpy()
        if hasattr(scores, "cpu")
        else np.asarray(scores, dtype=np.float32)
    )
    return (
        [_mask_to_numpy(m) for m in masks],
        [float(s) for s in scores_np],
    )


def _match_masks_to_boxes(
    candidate_masks: list[np.ndarray],
    candidate_scores: list[float],
    target_boxes: list[np.ndarray],
    empty_shape: tuple[int, int],
) -> list[tuple[np.ndarray, float]]:
    """Assign masks via mutual nearest-neighbor on mask-box IoU (bidirectional match)."""
    n_obj = len(target_boxes)
    if n_obj == 0:
        return []

    empty_mask = np.zeros(empty_shape, dtype=np.float32)
    if not candidate_masks:
        return [(empty_mask, 0.0)] * n_obj

    n_mask = len(candidate_masks)
    sim = np.zeros((n_obj, n_mask), dtype=np.float64)
    for i, box in enumerate(target_boxes):
        for j, mask in enumerate(candidate_masks):
            sim[i, j] = _mask_box_iou(mask, box)

    obj_best_mask = sim.argmax(axis=1)
    mask_best_obj = sim.argmax(axis=0)

    results: list[tuple[np.ndarray, float]] = [(empty_mask.copy(), 0.0)] * n_obj
    for i in range(n_obj):
        j = int(obj_best_mask[i])
        if sim[i, j] <= 0.0:
            continue
        if int(mask_best_obj[j]) != i:
            continue
        results[i] = (candidate_masks[j], candidate_scores[j])
    return results


def _infer_refiner(
    processor,
    model,
    images,
    boxes_per_image,
    tags_per_image,
    prompt_batch_size: int = 32,
    score_threshold: float = 0.5,
):
    """Text-only SAM3 refinement: one forward row per (image, unique tag)."""
    per_image: list[tuple[list, list]] = []
    tags_by_image: list[list[str]] = []
    queries: list[dict] = []

    for img_idx, image in enumerate(images):
        boxes = np.asarray(boxes_per_image[img_idx], dtype=np.float32).reshape(-1, 4)
        n = len(boxes)
        if n == 0:
            per_image.append(([], []))
            tags_by_image.append([])
            continue

        tags = _tags_per_box(
            tags_per_image[img_idx] if tags_per_image else None, n,
        )
        tags_by_image.append(tags)
        per_image.append((
            [np.zeros((1, 1), dtype=np.float32) for _ in range(n)],
            [0.0] * n,
        ))

        for tag in _unique_tags_preserve_order(tags):
            queries.append({"img_idx": img_idx, "tag": tag, "image": image})

    if not queries:
        return per_image

    device = next(model.parameters()).device
    batch_size = max(1, int(prompt_batch_size))
    for start in range(0, len(queries), batch_size):
        chunk = queries[start : start + batch_size]
        inputs = processor(
            images=[q["image"] for q in chunk],
            text=[q["tag"] for q in chunk],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        seg_list = _post_process(
            processor, outputs, score_threshold, _target_sizes(inputs),
        )

        for seg, q in zip(seg_list, chunk):
            img_idx = q["img_idx"]
            tag = q["tag"]
            tags = tags_by_image[img_idx]
            boxes = np.asarray(boxes_per_image[img_idx], dtype=np.float32).reshape(-1, 4)

            obj_indices = [i for i, t in enumerate(tags) if t == tag]
            target_boxes = [boxes[i] for i in obj_indices]
            cand_masks, cand_scores = _extract_seg_masks_scores(seg)

            h, w = images[img_idx].size[1], images[img_idx].size[0]
            if cand_masks:
                h, w = cand_masks[0].shape
            assignments = _match_masks_to_boxes(
                cand_masks, cand_scores, target_boxes, empty_shape=(h, w),
            )

            masks_out, scores_out = per_image[img_idx]
            for obj_idx, (mask, score) in zip(obj_indices, assignments):
                masks_out[obj_idx] = mask
                scores_out[obj_idx] = score

    return per_image


class Sam3Refiner(BaseTask):
    """Refine filter-stage masks with Sam3Model text-only prompts (NPU-safe)."""

    MIN_SCORE = 0.5
    MIN_MASK_PIXELS = 20

    def __init__(self, args, device=None):
        super().__init__(args)
        segmenter_model = args.get("segmenter_model", "facebook/sam3")
        devices = _parse_devices(
            args.get("device") or device or _default_device()
        )
        self.min_score = float(args.get("min_score", self.MIN_SCORE))
        self.min_mask_pixels = int(args.get("min_mask_pixels", self.MIN_MASK_PIXELS))
        self.prompt_batch_size = int(args.get("prompt_batch_size", 32))

        load_kwargs = {}
        replicas_per_device = int(args.get("replicas_per_device", 1))
        if any(_is_npu_device(dev) for dev in devices) and replicas_per_device != 1:
            raise ValueError(
                "SAM3 NPU mode supports exactly one replica per NPU. "
                "Set replicas_per_device: 1 to avoid known AICPU errors."
            )

        self._replica_pool: queue.Queue = queue.Queue()
        for dev in devices:
            for _ in range(replicas_per_device):
                proc, model = _load_sam3_replica(segmenter_model, load_kwargs, dev)
                self._replica_pool.put((proc, model, dev))

        total_replicas = len(devices) * replicas_per_device
        if total_replicas > 1:
            self.use_multi_processing = True
            if not args.get("num_workers"):
                args["num_workers"] = total_replicas

        print(
            f"[Sam3Refiner] Sam3Model text-only on {','.join(devices)} | "
            f"replicas={total_replicas} prompt_batch_size={self.prompt_batch_size} "
            f"pipeline_batch_size={args.get('batch_size', 4)} "
            f"min_score={self.min_score:.2f} min_mask_pixels={self.min_mask_pixels}",
            flush=True,
        )

        assert "update_keys" in args, "update_keys must be specified in args."
        self.output_dir = os.path.join(self.args.get("output_dir"), self.args.get("file_name"))

    @staticmethod
    def _masks_to_bboxes(masks):
        boxes = []
        for mask in masks:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                boxes.append([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
            else:
                boxes.append([0, 0, 0, 0])
        return np.array(boxes)

    def _refine(self, images, masks_list, tags_list):
        boxes_per_image = [self._masks_to_bboxes(m) for m in masks_list]
        processor, model, dev = self._replica_pool.get()
        try:
            _set_npu_device(dev)
            return _infer_refiner(
                processor, model,
                images, boxes_per_image, tags_list,
                prompt_batch_size=self.prompt_batch_size,
                score_threshold=self.min_score,
            )
        finally:
            self._replica_pool.put((processor, model, dev))

    def _filter_masks(self, pred_masks, scores):
        refined, keep_indices = [], []
        for i, arr in enumerate(pred_masks):
            if float(scores[i]) < self.min_score:
                continue
            arr = np.asarray(arr) > 0
            if np.sum(arr) > self.min_mask_pixels:
                refined.append(arr)
                keep_indices.append(i)
        if not keep_indices:
            return [], [], []
        bboxes = self._masks_to_bboxes([m.astype(bool) for m in refined]).tolist()
        return refined, bboxes, keep_indices

    def _load_example(self, example):
        image = Image.open(example["image"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        coarse = [_load_coarse_mask(p) for p in example["masks"]]
        return image, coarse

    def _save_example(self, example, idx, refined, bboxes_2d, keep_indices):
        self._filter_by_keep_indices(example, keep_indices)
        mask_files = self._save_masks(
            refined, os.path.join(self.output_dir, "masks"), str(idx),
        )
        if len(mask_files) != len(example["obj_tags"]):
            return None
        if len(mask_files) != len(example["bboxes_3d_world_coords"]):
            return None
        example["masks"] = mask_files
        example["bboxes_2d"] = bboxes_2d
        return example

    def _process_batch(self, batch_items):
        valid_items = []
        for idx, example in batch_items:
            try:
                self.validate_example(example)
                image, coarse = self._load_example(example)
                valid_items.append((idx, example, image, coarse))
            except Exception:
                pass

        if not valid_items:
            return [], {"samples": len(batch_items), "boxes_in": 0, "boxes_kept": 0, "samples_saved": 0}

        per_image = self._refine(
            [x[2] for x in valid_items],
            [x[3] for x in valid_items],
            [x[1].get("obj_tags") for x in valid_items],
        )
        boxes_in = sum(len(x[3]) for x in valid_items)
        boxes_kept = 0
        outputs = []

        for (idx, example, _, _), (pred_masks, scores) in zip(valid_items, per_image):
            refined, bboxes_2d, keep_indices = self._filter_masks(pred_masks, scores)
            boxes_kept += len(keep_indices)
            if not keep_indices:
                continue
            saved = self._save_example(example, idx, refined, bboxes_2d, keep_indices)
            if saved is not None:
                outputs.append(saved)

        return outputs, {
            "samples": len(batch_items),
            "boxes_in": boxes_in,
            "boxes_kept": boxes_kept,
            "samples_saved": len(outputs),
        }

    def _run_batched(self, dataset):
        num_workers = self.args.get("num_workers", 4)
        batch_size = int(self.args.get("batch_size", 4))
        examples = list(enumerate(dataset.to_dict("records")))
        batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]
        window = num_workers * 2

        processed = []
        total_samples = total_boxes_in = total_boxes_kept = total_samples_saved = 0
        next_log_at = 1000
        lock = threading.Lock()

        def _log():
            nonlocal next_log_at
            while total_samples >= next_log_at:
                print(
                    f"[Sam3Refiner] {total_samples} samples | "
                    f"bbox in={total_boxes_in} filtered={total_boxes_in - total_boxes_kept} "
                    f"kept={total_boxes_kept} | samples saved={total_samples_saved}",
                    flush=True,
                )
                next_log_at += 1000

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pbar = tqdm.tqdm(total=len(examples), desc="SAM3 samples")
            for chunk_start in range(0, len(batches), window):
                chunk = batches[chunk_start : chunk_start + window]
                futures = [executor.submit(self._process_batch, b) for b in chunk]
                for future in as_completed(futures):
                    try:
                        outs, st = future.result()
                        processed.extend(outs)
                        with lock:
                            total_samples += st["samples"]
                            total_boxes_in += st["boxes_in"]
                            total_boxes_kept += st["boxes_kept"]
                            total_samples_saved += st["samples_saved"]
                            pbar.update(st["samples"])
                            _log()
                    except Exception as exc:
                        import traceback as _tb
                        print(f"[WARN] SAM3 batch failed: {exc}")
                        print(_tb.format_exc())
            pbar.close()

        if total_samples > 0 and total_samples % 1000 != 0:
            print(
                f"[Sam3Refiner] {total_samples} samples (final) | "
                f"bbox in={total_boxes_in} filtered={total_boxes_in - total_boxes_kept} "
                f"kept={total_boxes_kept} | samples saved={total_samples_saved}",
                flush=True,
            )
        return pd.DataFrame(processed).reset_index(drop=True) if processed else pd.DataFrame()

    def run(self, dataset):
        if self.use_multi_processing:
            return self._run_batched(dataset)
        return super().run(dataset)

    def _save_masks(self, masks, mask_dir, prefix):
        os.makedirs(mask_dir, exist_ok=True)
        paths = []
        for i, mask in enumerate(masks):
            path = os.path.join(mask_dir, f"example_{prefix}_box_{i}_mask.png")
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)
            paths.append(path)
        return paths

    def validate_example(self, example):
        for key in ("image", "masks", "obj_tags"):
            if key not in example:
                raise ValueError(f"{key} not found in example")
        if len(example["obj_tags"]) == 0:
            raise ValueError("obj_tags is empty")

    def _filter_by_keep_indices(self, example, keep_indices):
        for key in self.args.get("update_keys", []):
            example[key] = [example[key][i] for i in keep_indices]
        return example

    def apply_transform(self, example, idx):
        self.validate_example(example)
        image, coarse = self._load_example(example)
        pred_masks, scores = self._refine([image], [coarse], [example.get("obj_tags")])[0]
        refined, bboxes_2d, keep_indices = self._filter_masks(pred_masks, scores)
        if not keep_indices:
            return None, False

        saved = self._save_example(example, idx, refined, bboxes_2d, keep_indices)
        assert saved is not None
        return saved, True
