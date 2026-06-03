import numpy as np
import torch
import os
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import Sam3TrackerProcessor, Sam3TrackerModel

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_utils import merge_overlapping_masks, merge_overlapping_boxes
from task.base_task import BaseTask


class Localizer(BaseTask):
    """Grounding DINO + SAM3 Tracker pipeline: detect objects and generate segmentation masks.

    Replaces the SAM2 segmenter with SAM3 Tracker loaded via the transformers library,
    preserving all detection, merging, and mask-saving logic from the SAM2 version.
    Recommended: transformers >= 5.0.0
    """

    def __init__(self, args, device=None):
        super().__init__(args)
        grounding_model = args.get("grounding_model", "IDEA-Research/grounding-dino-tiny")
        segmenter_model = args.get("segmenter_model", "facebook/sam3")
        device = args.get("device") or device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Grounding DINO for open-vocabulary detection (unchanged)
        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(device)

        # SAM3 Tracker for segmentation via transformers (replaces SAM2ImagePredictor)
        self.seg_processor = Sam3TrackerProcessor.from_pretrained(segmenter_model)
        self.seg_model = Sam3TrackerModel.from_pretrained(segmenter_model).to(device)
        self.seg_model.eval()

        self.output_dir = args.get("output_dir")

    def _load_image(self, img):
        """Load and convert an image input to RGB PIL Image."""
        if isinstance(img, list):
            img = img[0]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def detect_and_segment(self, image, obj_tags):
        """Run grounding detection + SAM3 Tracker segmentation, filter and merge results.

        Args:
            image: RGB PIL Image.
            obj_tags: list of object tag strings to detect.

        Returns:
            (masks, boxes, tags) on success, or None if fewer than 2 objects found.
        """
        text_prompt = ". ".join(obj_tags)

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        det_tags = results[0]["text_labels"]
        det_boxes = results[0]["boxes"].cpu().numpy()

        # Need at least 2 distinct objects
        if len(det_tags) <= 1:
            return None

        # Merge overlapping boxes with same tag
        det_boxes, det_tags = merge_overlapping_boxes(det_tags, det_boxes, overlap_threshold=0.8)

        # SAM3 Tracker: pass detected boxes as per-object prompts.
        # input_boxes format: [batch=1, num_objects=N, 4]
        seg_inputs = self.seg_processor(
            images=image,
            input_boxes=[det_boxes.tolist()],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            seg_outputs = self.seg_model(**seg_inputs, multimask_output=False)

        # post_process_masks → (N, 1, H, W) tensor; iou_scores → (1, N, 1)
        pred_masks = self.seg_processor.post_process_masks(
            seg_outputs.pred_masks.cpu(),
            seg_inputs["original_sizes"],
        )[0]  # (N, 1, H, W) tensor
        scores = seg_outputs.iou_scores[0, :, 0].cpu().numpy()  # (N,)

        # Filter by confidence score
        keep = [i for i, score in enumerate(scores) if score >= 0.7]
        if len(keep) == 0:
            return None

        masks = pred_masks[keep].numpy()  # (keep, 1, H, W) numpy array
        det_tags = [det_tags[i] for i in keep]
        det_boxes = det_boxes[keep]

        # Merge overlapping masks with same tag
        masks, det_tags, det_boxes = merge_overlapping_masks(
            masks, det_tags, det_boxes, overlap_threshold=0.8
        )

        # Require at least 2 distinct objects after merging
        if len(det_tags) <= 1:
            return None

        return masks, det_boxes.tolist(), det_tags

    def _save_masks(self, masks, mask_dir, prefix):
        """Save binary masks as PNG files.

        Args:
            masks: array of shape (N, 1, H, W) — raw SAM3 Tracker output masks.
            mask_dir: directory to save mask images.
            prefix: filename prefix (e.g. "mask_3" or "mask_3_0").

        Returns:
            list of saved file paths.
        """
        os.makedirs(mask_dir, exist_ok=True)
        file_list = []
        for i, mask in enumerate(masks):
            binary = (mask[0] > 0).astype(np.uint8) * 255
            mask_image = Image.fromarray(binary, mode='L')
            path = os.path.join(mask_dir, f"mask_{prefix}_{i}.png")
            mask_image.save(path, format='PNG')
            file_list.append(path)
        return file_list

    def apply_transform(self, example, idx):
        """Detect and segment objects, save masks, update example fields.

        Populates:
            example["masks"]: list of mask file paths.
            example["bboxes_2d"]: list of [x1, y1, x2, y2] bounding boxes.
            example["obj_tags"]: list of detected object tag strings.
        """
        img_idx = str(idx)
        mask_dir = os.path.join(self.output_dir, self.args.get("file_name"), "masks")

        is_batched = (isinstance(example["image"], list)
                      and isinstance(example["image"][0], (list, Image.Image)))

        if is_batched:
            all_valid = True
            all_masks, all_boxes, all_tags = [], [], []
            for i, img_item in enumerate(example["image"]):
                image = self._load_image(img_item)
                tags = example["obj_tags"][i]
                result = self.detect_and_segment(image, tags)
                if result is None:
                    all_valid = False
                    all_masks.append([])
                    all_boxes.append([])
                    all_tags.append([])
                else:
                    masks, boxes, det_tags = result
                    mask_files = self._save_masks(masks, mask_dir, f"{img_idx}_{i}")
                    all_masks.append(mask_files)
                    all_boxes.append(boxes)
                    all_tags.append(det_tags)

            example["masks"] = all_masks
            example["bboxes_2d"] = all_boxes
            example["obj_tags"] = all_tags
            return example, all_valid
        else:
            image = self._load_image(example["image"])
            result = self.detect_and_segment(image, example["obj_tags"])
            if result is None:
                return example, False

            masks, boxes, det_tags = result
            mask_files = self._save_masks(masks, mask_dir, img_idx)

            assert len(mask_files) == len(boxes) == len(det_tags), (
                f"Length mismatch: {len(mask_files)} masks, "
                f"{len(boxes)} boxes, {len(det_tags)} tags."
            )

            example["masks"] = mask_files
            example["bboxes_2d"] = boxes
            example["obj_tags"] = det_tags
            return example, True
