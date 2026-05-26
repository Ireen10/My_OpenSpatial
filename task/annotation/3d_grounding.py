"""
3D grounding annotation task: predict 3D bounding boxes for objects.

Sub-tasks:
    grounding_oe — given object names, predict their 3D bounding boxes (open-ended).

Coordinate system:
    3D boxes are in camera coordinates, converted from world-frame boxes
    via SceneNode.box_3d_in_camera().

    Box format: [x, y, z, xl, yl, zl, roll, pitch, yaw]
    Euler angles in radians, rotation order zxy.

Templates used:
    grounding_3d.open_ended — introduction (camera) + stem + question_instruction (JSON)
"""

import random

from .core.base_annotation_task import BaseAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes
from utils.box_utils import format_bbox_3d_for_grounding, format_grounding_answer_json


class ThreeDGroundingGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Singleview 3D Grounding"
    SUB_TASKS = {
        "grounding_oe": {"default": 1, "handler": "_generate_grounding_oe"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.task_name = args.get("task_name") or args.get("file_name", "3d_grounding")

    def check_example(self, example) -> bool:
        return super().check_example(example)

    @staticmethod
    def _camera_shared(intrinsic, img_dim) -> dict:
        w, h = int(img_dim[0]), int(img_dim[1])
        return {
            "FX": f"{float(intrinsic[0, 0]):.2f}",
            "FY": f"{float(intrinsic[1, 1]):.2f}",
            "CX": f"{float(intrinsic[0, 2]):.2f}",
            "CY": f"{float(intrinsic[1, 2]):.2f}",
            "W": str(w),
            "H": str(h),
        }

    def grounding_oe_prompt_func(self, sampled_tags, tags_to_boxes, camera_shared):
        """Generate an open-ended 3D grounding QA."""
        entries = []
        for tag in sampled_tags:
            for box in tags_to_boxes[tag]:
                bbox = format_bbox_3d_for_grounding(box)
                if bbox is not None:
                    entries.append({"label": tag, "bbox_3d": bbox})
        if not entries:
            return None
        return self.render_structured_prompt(
            "grounding_3d.open_ended",
            shared={
                **camera_shared,
                "A": ", ".join(sampled_tags),
                "X": format_grounding_answer_json(entries),
            },
        )

    def _generate_grounding_oe(self, graph):
        """Sample 1-3 object tags and ask for their 3D bounding boxes."""
        view = graph.primary_view
        pose = view.pose
        if pose is None:
            return None

        depth_map = view.depth_map
        img_dim = depth_map.shape[::-1] if depth_map is not None else view.image.size
        camera_shared = self._camera_shared(view.intrinsic, img_dim)

        nodes = graph.get_object_nodes()
        if self.filter_tags is not None:
            nodes = [n for n in nodes if n.tag not in self.filter_tags]

        tags_to_boxes = {}
        for node in nodes:
            cam_box = node.box_3d_in_camera(pose)
            if cam_box is not None:
                tags_to_boxes.setdefault(node.tag, []).append(cam_box)
        if not tags_to_boxes:
            return None

        unique_tags = list(tags_to_boxes.keys())
        sampled_tags = random.sample(unique_tags, random.randint(1, min(3, len(unique_tags))))

        prompt = self.grounding_oe_prompt_func(sampled_tags, tags_to_boxes, camera_shared)
        if prompt is None:
            return None

        extra_slots = {}
        for i, tag in enumerate(sampled_tags):
            sid = chr(ord("A") + i)
            node = next((n for n in nodes if n.tag == tag), None)
            oid = 0
            if node is not None:
                try:
                    oid = int(node.node_id)
                except (TypeError, ValueError):
                    oid = i
            extra_slots[sid] = {"obj_idx": oid, "tag": tag}
        self._record_turn(
            "grounding_oe",
            "grounding_3d.open_ended",
            prompt,
            QuestionType.OPEN_ENDED,
            mark_spec=None,
            extra_slots=extra_slots,
        )
        return prompt, {"bytes": convert_pil_to_bytes(view.image)}, QuestionType.OPEN_ENDED
