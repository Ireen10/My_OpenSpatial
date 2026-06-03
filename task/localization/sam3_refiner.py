import numpy as np
import torch
from PIL import Image
import os

from transformers import Sam3TrackerProcessor, Sam3TrackerModel
from task.base_task import BaseTask


class Sam3Refiner(BaseTask):
    """Refine coarse masks using SAM3 Tracker box-prompt segmentation (via transformers).

    Drop-in replacement for Sam2Refiner. Uses Sam3TrackerModel from the transformers
    library instead of the sam2 package, preserving all filtering and saving logic.
    Recommended: transformers >= 5.0.0
    """

    MIN_SCORE = 0.6
    MIN_MASK_PIXELS = 20

    def __init__(self, args, device=None):
        super().__init__(args)
        segmenter_model = args.get("segmenter_model", "facebook/sam3")
        device = args.get("device") or device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.processor = Sam3TrackerProcessor.from_pretrained(segmenter_model)
        self.model = Sam3TrackerModel.from_pretrained(segmenter_model).to(device)
        self.model.eval()

        assert "update_keys" in args, "update_keys must be specified in args."
        self.output_dir = os.path.join(self.args.get("output_dir"), self.args.get("file_name"))

    @staticmethod
    def _masks_to_bboxes(masks):
        """Compute axis-aligned bounding boxes from binary masks.

        Args:
            masks: list of 2D boolean/uint8 arrays.

        Returns:
            np.ndarray of shape (N, 4) with [x1, y1, x2, y2] per mask.
        """
        boxes = []
        for mask in masks:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                boxes.append([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
            else:
                boxes.append([0, 0, 0, 0])
        return np.array(boxes)

    @staticmethod
    def _squeeze_mask(arr):
        """Squeeze (1, H, W) → (H, W) if needed."""
        return arr[0] if arr.ndim == 3 else arr

    def refine_masks(self, image, masks):
        """Re-segment each mask region using SAM3 Tracker box prompts.

        Args:
            image: RGB PIL Image.
            masks: list of 2D mask arrays (coarse masks to refine).

        Returns:
            (refined_masks, bboxes_2d, keep_indices) where keep_indices
            maps back to the original mask list. Returns ([], [], []) if
            no valid masks survive filtering.
        """
        input_boxes_np = self._masks_to_bboxes(masks)

        # Sam3TrackerProcessor expects input_boxes as [batch, num_objects, 4]
        input_boxes_list = [input_boxes_np.tolist()]

        inputs = self.processor(
            images=image,
            input_boxes=input_boxes_list,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)

        # post_process_masks returns a list of tensors (one per image in batch).
        # Each tensor has shape (num_objects, num_candidates, H, W).
        # With multimask_output=False, num_candidates=1.
        pred_masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]  # (N, 1, H, W) tensor

        # iou_scores shape: (batch=1, num_objects, num_candidates=1)
        scores = outputs.iou_scores[0, :, 0].cpu().numpy()  # (N,)

        # Two-pass filtering: score threshold, then minimum pixel count
        refined, keep_indices = [], []
        for i, score in enumerate(scores):
            if score < self.MIN_SCORE:
                continue
            arr = self._squeeze_mask(pred_masks[i].numpy())  # (H, W) bool/float
            if np.sum(arr) > self.MIN_MASK_PIXELS:
                refined.append(arr)
                keep_indices.append(i)

        if not keep_indices:
            return [], [], []

        bboxes_2d = self._masks_to_bboxes([m.astype(bool) for m in refined]).tolist()
        return refined, bboxes_2d, keep_indices

    def _save_masks(self, masks, mask_dir, prefix):
        """Save binary masks as grayscale PNG files.

        Args:
            masks: list of 2D arrays (values treated as boolean).
            mask_dir: output directory.
            prefix: filename prefix for saved files.

        Returns:
            list of saved file paths.
        """
        os.makedirs(mask_dir, exist_ok=True)
        file_list = []
        for i, mask in enumerate(masks):
            binary = (mask * 255).astype(np.uint8)
            img = Image.fromarray(binary, mode='L')
            path = os.path.join(mask_dir, f"example_{prefix}_box_{i}_mask.png")
            img.save(path)
            file_list.append(path)
        return file_list

    def validate_example(self, example):
        """Check that required fields exist and are non-empty."""
        for key in ("image", "masks", "obj_tags"):
            if key not in example:
                raise ValueError(f"{key} not found in example")
        if len(example["obj_tags"]) == 0:
            raise ValueError("obj_tags is empty")

    def _filter_by_keep_indices(self, example, keep_indices):
        """Keep only elements at keep_indices for each field in update_keys."""
        update_keys = self.args.get("update_keys", [])
        if not update_keys or keep_indices is None:
            return example
        for key in update_keys:
            example[key] = [example[key][i] for i in keep_indices]
        return example

    def apply_transform(self, example, idx):
        """Refine masks, filter by quality, save results.

        Populates:
            example["masks"]: list of refined mask file paths.
            example["bboxes_2d"]: list of [x1, y1, x2, y2] bounding boxes.
        """
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
