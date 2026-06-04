import warnings
import numpy as np
import torch
import os
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_utils import merge_overlapping_masks, merge_overlapping_boxes
from task.base_task import BaseTask


def _validate_model_path(model_name_or_path: str) -> None:
    """Raise a clear FileNotFoundError when a local path does not exist."""
    import os as _os
    is_local = (
        _os.path.isabs(model_name_or_path)
        or model_name_or_path.startswith("./")
        or model_name_or_path.startswith("../")
    )
    if is_local and not _os.path.isdir(model_name_or_path):
        raise FileNotFoundError(
            f"SAM3 weights directory not found: {model_name_or_path!r}\n"
            "Check 'segmenter_model' in your YAML — must be an absolute local path "
            "or a HuggingFace Hub repo ID (e.g. 'facebook/sam3')."
        )


def _load_sam3_image_model(segmenter_model, torch_dtype, device):
    """Load SAM3 in single-image mode.

    Prefers Sam3Processor + Sam3Model (image predictor, no memory module) which
    produces standard segmentation-quality IoU scores.  Falls back to the Tracker
    variant if the image-predictor classes are not available in the installed
    transformers version.

    Using the Tracker variant (Sam3TrackerModel) for single-frame inference causes
    two problems:
      1. roi_align (used in the memory encoder) is not supported on Ascend NPU and
         falls back to CPU, causing dtype-mismatch artefacts that depress IoU scores.
      2. The tracker's IoU score measures temporal-tracking quality, not segmentation
         quality, so single-frame scores are systematically ~0.03–0.05 instead of the
         expected 0.7–0.99 range.
    """
    _validate_model_path(segmenter_model)
    load_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}
    suppress_ctx = warnings.catch_warnings()
    suppress_ctx.__enter__()
    warnings.filterwarnings(
        "ignore",
        message=r"You are using a model of type sam3_video",
        category=UserWarning,
    )

    try:
        from transformers import Sam3Processor, Sam3Model  # image-predictor variant
        proc = Sam3Processor.from_pretrained(segmenter_model)
        model = Sam3Model.from_pretrained(segmenter_model, **load_kwargs).to(device)
        suppress_ctx.__exit__(None, None, None)
        return proc, model, "Sam3Model"
    except (ImportError, AttributeError, OSError, ValueError):
        # OSError/ValueError: model_type mismatch with Sam3Model (sam3_video config)
        # — fall back to tracker variant which accepts the shared checkpoint.
        pass

    # Fallback: tracker variant (acceptable if transformers < version with Sam3Model)
    from transformers import Sam3TrackerProcessor, Sam3TrackerModel
    proc = Sam3TrackerProcessor.from_pretrained(segmenter_model)
    model = Sam3TrackerModel.from_pretrained(segmenter_model, **load_kwargs).to(device)
    suppress_ctx.__exit__(None, None, None)
    return proc, model, "Sam3TrackerModel (fallback)"


def _post_process_pred_masks(processor, pred_masks, original_sizes):
    """Sam3TrackerProcessor vs Sam3Processor mask upscaling (see sam3_refiner)."""
    masks = pred_masks.cpu() if hasattr(pred_masks, "cpu") else pred_masks
    if hasattr(processor, "post_process_masks"):
        return processor.post_process_masks(masks, original_sizes)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, "post_process_masks"):
        return image_processor.post_process_masks(masks, original_sizes)
    raise AttributeError(
        f"{type(processor).__name__} has no post_process_masks."
    )


class Localizer(BaseTask):
    """Grounding DINO + SAM3 pipeline: detect objects and generate segmentation masks.

    Uses Sam3Model (image-predictor variant) for segmentation to avoid the
    roi_align NPU fallback and tracker-specific IoU score semantics.
    Recommended: transformers >= 5.0.0
    """

    def __init__(self, args, device=None):
        super().__init__(args)
        grounding_model = args.get("grounding_model", "IDEA-Research/grounding-dino-tiny")
        segmenter_model = args.get("segmenter_model", "facebook/sam3")
        device = args.get("device") or device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        torch_dtype_str = args.get("torch_dtype", None)
        torch_dtype = getattr(torch, torch_dtype_str) if torch_dtype_str else None

        # Grounding DINO for open-vocabulary detection (unchanged)
        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(device)

        # SAM3 image predictor — loads Sam3Model if available, else Sam3TrackerModel
        self.seg_processor, self.seg_model, _variant = _load_sam3_image_model(
            segmenter_model, torch_dtype, device
        )
        print(f"[Localizer] SAM3 loaded as {_variant} on {device}")
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
        pred_masks = _post_process_pred_masks(
            self.seg_processor, seg_outputs.pred_masks, seg_inputs["original_sizes"]
        )[0]  # (N, 1, H, W) tensor
        scores = seg_outputs.iou_scores[0, :, 0].cpu().numpy()  # (N,)

        # Filter by confidence score (same threshold as Sam3Refiner.MIN_SCORE)
        keep = [i for i, score in enumerate(scores) if score >= 0.6]
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
