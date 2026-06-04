import numpy as np
import torch
import os
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_utils import merge_overlapping_masks, merge_overlapping_boxes
from task.base_task import BaseTask
from task.localization.sam3_refiner import _load_sam3_replica, _target_sizes, _post_process


def _segment_boxes(processor, model, image, boxes_np, device, min_score: float):
    """Sam3Model: multi-box forward + post_process (used by verify / Localizer)."""
    inputs = processor(
        images=image,
        input_boxes=[boxes_np.tolist()],
        input_boxes_labels=[[1] * len(boxes_np)],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    seg = _post_process(processor, outputs, min_score, _target_sizes(inputs))[0]
    masks = seg["masks"]
    if hasattr(masks, "float"):
        masks = masks.float()
    scores = seg["scores"].float().cpu().numpy()
    return masks, scores


class Localizer(BaseTask):
    """Grounding DINO + Sam3Model. Requires transformers >= 5.0.0."""

    def __init__(self, args, device=None):
        super().__init__(args)
        grounding_model = args.get("grounding_model", "IDEA-Research/grounding-dino-tiny")
        segmenter_model = args.get("segmenter_model", "facebook/sam3")
        device = args.get("device") or device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(device)
        self.seg_processor, self.seg_model = _load_sam3_replica(segmenter_model, {}, device)
        print(f"[Localizer] Sam3Model on {device}")
        self.seg_model.eval()
        self.output_dir = args.get("output_dir")

    def _load_image(self, img):
        if isinstance(img, list):
            img = img[0]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def detect_and_segment(self, image, obj_tags):
        text_prompt = ". ".join(obj_tags)
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )
        det_tags = results[0]["text_labels"]
        det_boxes = results[0]["boxes"].cpu().numpy()

        if len(det_tags) <= 1:
            return None

        det_boxes, det_tags = merge_overlapping_boxes(det_tags, det_boxes, overlap_threshold=0.8)

        pred_masks, scores = _segment_boxes(
            self.seg_processor, self.seg_model, image, det_boxes, self.device, min_score=0.6
        )
        if len(scores) == 0:
            return None

        pm = pred_masks
        if hasattr(pm, "float"):
            pm = pm.float()
        masks = pm.cpu().numpy() if hasattr(pm, "cpu") else np.asarray(pm)
        n = min(len(masks), len(det_tags), len(det_boxes))
        masks, det_tags, det_boxes = masks[:n], det_tags[:n], det_boxes[:n]

        masks, det_tags, det_boxes = merge_overlapping_masks(
            masks, det_tags, det_boxes, overlap_threshold=0.8
        )
        if len(det_tags) <= 1:
            return None
        return masks, det_boxes.tolist(), det_tags

    def _save_masks(self, masks, mask_dir, prefix):
        os.makedirs(mask_dir, exist_ok=True)
        paths = []
        for i, mask in enumerate(masks):
            binary = (mask[0] > 0).astype(np.uint8) * 255
            path = os.path.join(mask_dir, f"mask_{prefix}_{i}.png")
            Image.fromarray(binary, mode="L").save(path, format="PNG")
            paths.append(path)
        return paths

    def apply_transform(self, example, idx):
        img_idx = str(idx)
        mask_dir = os.path.join(self.output_dir, self.args.get("file_name"), "masks")
        is_batched = (
            isinstance(example["image"], list)
            and isinstance(example["image"][0], (list, Image.Image))
        )

        if is_batched:
            all_valid = True
            all_masks, all_boxes, all_tags = [], [], []
            for i, img_item in enumerate(example["image"]):
                image = self._load_image(img_item)
                result = self.detect_and_segment(image, example["obj_tags"][i])
                if result is None:
                    all_valid = False
                    all_masks.append([])
                    all_boxes.append([])
                    all_tags.append([])
                else:
                    masks, boxes, det_tags = result
                    all_masks.append(self._save_masks(masks, mask_dir, f"{img_idx}_{i}"))
                    all_boxes.append(boxes)
                    all_tags.append(det_tags)
            example["masks"] = all_masks
            example["bboxes_2d"] = all_boxes
            example["obj_tags"] = all_tags
            return example, all_valid

        image = self._load_image(example["image"])
        result = self.detect_and_segment(image, example["obj_tags"])
        if result is None:
            return example, False

        masks, boxes, det_tags = result
        mask_files = self._save_masks(masks, mask_dir, img_idx)
        assert len(mask_files) == len(boxes) == len(det_tags)
        example["masks"] = mask_files
        example["bboxes_2d"] = boxes
        example["obj_tags"] = det_tags
        return example, True
