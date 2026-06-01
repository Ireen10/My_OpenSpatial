import os
import sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from task.base_task import BaseTask

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None  # type: ignore

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.data_utils import merge_overlapping_masks, merge_overlapping_boxes


class Localizer(BaseTask):
    """Grounding DINO + SAM3 pipeline: detect objects and generate segmentation masks."""

    def __init__(self, args, device=None):
        super().__init__(args)
        grounding_model = args.get("grounding_model", "IDEA-Research/grounding-dino-tiny")
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

        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(
            self.device
        )

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
            confidence_threshold=0.7,
        )

        self.output_dir = args.get("output_dir")
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

    def _load_image(self, img):
        if isinstance(img, list):
            img = img[0]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _sam3_masks_from_boxes(self, image, det_boxes):
        image_w, image_h = image.size
        base_state = self.sam3_processor.set_image(image, state={})
        masks, scores = [], []

        for box_xyxy in det_boxes:
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
            best_score = float(pred_scores[best_idx].item())
            best_mask = pred_masks[best_idx]
            if best_mask.ndim == 2:
                best_mask = best_mask.unsqueeze(0)
            masks.append(best_mask.detach().cpu().numpy().astype(np.uint8))
            scores.append(best_score)

        if len(masks) == 0:
            return None, None
        return np.stack(masks, axis=0), np.array(scores)

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
        masks, scores = self._sam3_masks_from_boxes(image, det_boxes)
        if masks is None or scores is None:
            return None

        valid_len = min(len(det_tags), len(det_boxes), len(scores), len(masks))
        det_tags = det_tags[:valid_len]
        det_boxes = det_boxes[:valid_len]
        masks = masks[:valid_len]
        scores = scores[:valid_len]

        keep = [i for i, score in enumerate(scores) if score >= 0.7]
        if len(keep) == 0:
            return None

        masks = masks[keep]
        det_tags = [det_tags[i] for i in keep]
        det_boxes = det_boxes[keep]

        masks, det_tags, det_boxes = merge_overlapping_masks(
            masks, det_tags, det_boxes, overlap_threshold=0.8
        )

        if len(det_tags) <= 1:
            return None

        return masks, det_boxes.tolist(), det_tags

    def _save_masks(self, masks, mask_dir, prefix):
        os.makedirs(mask_dir, exist_ok=True)
        file_list = []
        for i, mask in enumerate(masks):
            binary = (mask[0] > 0).astype(np.uint8) * 255
            mask_image = Image.fromarray(binary, mode="L")
            path = os.path.join(mask_dir, f"mask_{prefix}_{i}.png")
            mask_image.save(path, format="PNG")
            file_list.append(path)
        return file_list

    def apply_transform(self, example, idx):
        img_idx = str(idx)
        mask_dir = os.path.join(self.output_dir, self.args.get("file_name"), "masks")

        is_batched = isinstance(example["image"], list) and isinstance(
            example["image"][0], (list, Image.Image)
        )

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
