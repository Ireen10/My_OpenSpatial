import os
import numpy as np
import torch
from PIL import Image

from task.base_task import BaseTask

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None  # type: ignore

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class Sam3Refiner(BaseTask):
    """Refine coarse masks using SAM3 geometric box prompts."""

    MIN_SCORE = 0.6
    MIN_MASK_PIXELS = 20

    def __init__(self, args, device=None):
        super().__init__(args)
        self.device = self._resolve_device(args, device)
        self._configure_hf_cache(args)
        self._prepare_npu_runtime_if_needed()

        segmenter_model = args.get("segmenter_model", "facebook/sam3")
        segmenter_checkpoint_path = args.get("segmenter_checkpoint_path")
        segmenter_bpe_path = args.get("segmenter_bpe_path")
        segmenter_load_from_hf = args.get("segmenter_load_from_hf", True)
        segmenter_resolution = args.get("segmenter_resolution", 1008)

        if not segmenter_load_from_hf and not segmenter_checkpoint_path:
            raise ValueError(
                "segmenter_load_from_hf is False but segmenter_checkpoint_path is not provided."
            )

        # SAM3 official builder currently only moves model when device == 'cuda'.
        # Build on CPU first, then move to target device (including NPU).
        self.sam3_model = build_sam3_image_model(
            bpe_path=segmenter_bpe_path,
            device="cpu",
            eval_mode=True,
            checkpoint_path=segmenter_checkpoint_path,
            load_from_HF=segmenter_load_from_hf,
        ).to(self.device)
        self.sam3_model.eval()

        self.sam3_processor = Sam3Processor(
            self.sam3_model,
            resolution=segmenter_resolution,
            device=self.device,
            confidence_threshold=self.MIN_SCORE,
        )

        assert "update_keys" in args, "update_keys must be specified in args."
        self.output_dir = os.path.join(self.args.get("output_dir"), self.args.get("file_name"))
        self.segmenter_model = segmenter_model

    @staticmethod
    def _resolve_device(args, device):
        explicit = args.get("device") or device
        if explicit:
            return explicit
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu:0"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _configure_hf_cache(args):
        hf_home = args.get("hf_home")
        hf_hub_cache = args.get("hf_hub_cache")
        if hf_home:
            os.environ["HF_HOME"] = hf_home
        if hf_hub_cache:
            os.environ["HF_HUB_CACHE"] = hf_hub_cache

    def _prepare_npu_runtime_if_needed(self):
        if isinstance(self.device, str) and self.device.startswith("npu"):
            if not hasattr(torch, "npu"):
                raise RuntimeError(
                    "NPU device requested but torch.npu is unavailable. "
                    "Please install/enable torch_npu for Ascend runtime."
                )
            torch.npu.set_device(self.device)

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

    @staticmethod
    def _xyxy_to_norm_cxcywh(box_xyxy, image_w, image_h):
        x1, y1, x2, y2 = box_xyxy
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        cx = float(x1) + w / 2.0
        cy = float(y1) + h / 2.0
        if image_w <= 0 or image_h <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [cx / image_w, cy / image_h, w / image_w, h / image_h]

    @staticmethod
    def _build_prompt_state(base_state):
        return {
            "backbone_out": base_state["backbone_out"],
            "original_height": base_state["original_height"],
            "original_width": base_state["original_width"],
        }

    def refine_masks(self, image, masks):
        image_w, image_h = image.size
        input_boxes = self._masks_to_bboxes(masks)
        base_state = self.sam3_processor.set_image(image, state={})

        refined, bboxes_2d, keep_indices = [], [], []
        for i, box_xyxy in enumerate(input_boxes):
            if (box_xyxy[2] <= box_xyxy[0]) or (box_xyxy[3] <= box_xyxy[1]):
                continue

            box_cxcywh_norm = self._xyxy_to_norm_cxcywh(box_xyxy, image_w, image_h)
            state = self._build_prompt_state(base_state)
            state = self.sam3_processor.add_geometric_prompt(
                box=box_cxcywh_norm,
                label=True,
                state=state,
            )

            pred_masks = state.get("masks")
            pred_scores = state.get("scores")
            if pred_masks is None or pred_scores is None or len(pred_scores) == 0:
                continue

            best_idx = int(torch.argmax(pred_scores).item())
            score = float(pred_scores[best_idx].item())
            if score < self.MIN_SCORE:
                continue

            mask = pred_masks[best_idx]
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            mask_np = mask.detach().cpu().numpy().astype(np.uint8)
            if np.sum(mask_np) <= self.MIN_MASK_PIXELS:
                continue

            refined.append(mask_np)
            bboxes_2d.append(
                [int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])]
            )
            keep_indices.append(i)

        if not keep_indices:
            return [], [], []
        return refined, bboxes_2d, keep_indices

    def _save_masks(self, masks, mask_dir, prefix):
        os.makedirs(mask_dir, exist_ok=True)
        file_list = []
        for i, mask in enumerate(masks):
            binary = (mask > 0).astype(np.uint8) * 255
            img = Image.fromarray(binary, mode="L")
            path = os.path.join(mask_dir, f"example_{prefix}_box_{i}_mask.png")
            img.save(path)
            file_list.append(path)
        return file_list

    def validate_example(self, example):
        for key in ("image", "masks", "obj_tags"):
            if key not in example:
                raise ValueError(f"{key} not found in example")
        if len(example["obj_tags"]) == 0:
            raise ValueError("obj_tags is empty")

    def _filter_by_keep_indices(self, example, keep_indices):
        update_keys = self.args.get("update_keys", [])
        if not update_keys or keep_indices is None:
            return example
        for key in update_keys:
            example[key] = [example[key][i] for i in keep_indices]
        return example

    def apply_transform(self, example, idx):
        self.validate_example(example)

        image = Image.open(example["image"])
        if image.mode != "RGB":
            image = image.convert("RGB")

        coarse_masks = [np.array(Image.open(p)) for p in example["masks"]]
        refined_masks, bboxes_2d, keep_indices = self.refine_masks(image, coarse_masks)

        if not keep_indices:
            return None, False

        self._filter_by_keep_indices(example, keep_indices)

        mask_dir = os.path.join(self.output_dir, "masks")
        mask_files = self._save_masks(refined_masks, mask_dir, prefix=str(idx))

        assert len(mask_files) == len(example["obj_tags"]), (
            f"Mask count ({len(mask_files)}) != obj_tags count ({len(example['obj_tags'])})"
        )
        assert len(mask_files) == len(example["bboxes_3d_world_coords"]), (
            f"Mask count ({len(mask_files)}) != bboxes_3d count ({len(example['bboxes_3d_world_coords'])})"
        )

        example["masks"] = mask_files
        example["bboxes_2d"] = bboxes_2d
        return example, True
