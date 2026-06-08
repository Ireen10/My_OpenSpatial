"""
Size annotation task: absolute size measurement & relative size comparison.

Templates:
    size.absolute.single_view.{cm,m} / size.height.single_view.{cm,m} — OE + unit instruction
    size.big.single_view / size.small.single_view — Judgment, unconstrained instruction
"""

import random
from .core.base_annotation_task import BaseAnnotationTask
from .core.sample_metadata import marked_surface_label
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType
from utils.box_utils import RELATIVE_SIZE_DIAG_RATIO_MIN, box_3d_diag_extent

from .metric_gating import ABSOLUTE_DISTANCE_MODES, format_distance_value, pick_instruction_mode


class AnnotationGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Singleview Size"
    SUB_TASKS = {
        "absolute_size":   {"default": 1, "handler": "_generate_absolute_size"},
        "relative_size":   {"default": 1, "handler": "_generate_relative_size"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.task_name = args.get("task_name") or args.get("file_name", "size")

    def get_mark_config(self):
        return MarkConfig(mark_types=["box", "point"], shuffle_colors=True)

    def _get_node_extent(self, node):
        if node.box_3d_world is not None:
            return node.box_3d_world[3:6]
        cloud = node.view_appearances[0].pointcloud_camera
        return cloud.get_axis_aligned_bounding_box().get_extent()

    def relative_size_prompt_func(self, A, B):
        A_desc = marked_surface_label(A)
        B_desc = marked_surface_label(B)
        _, A_node = A
        _, B_node = B

        d_A = box_3d_diag_extent(A_node.box_3d_world)
        d_B = box_3d_diag_extent(B_node.box_3d_world)
        lo, hi = min(d_A, d_B), max(d_A, d_B)
        if lo <= 0 or hi / lo < RELATIVE_SIZE_DIAG_RATIO_MIN:
            return None, None

        if random.random() < 0.5:
            tpl = "size.big.single_view"
            prompt = self.render_structured_prompt(
                tpl, condition=d_A > d_B, shared={"A": A_desc, "B": B_desc},
            )
        else:
            tpl = "size.small.single_view"
            prompt = self.render_structured_prompt(
                tpl, condition=d_A < d_B, shared={"A": A_desc, "B": B_desc},
            )
        return prompt, tpl

    def absolute_size_prompt_func(self, marked, stem_kind, get_value):
        A_desc = marked_surface_label(marked)
        _, node = marked
        value_m = float(get_value(node))
        mode = pick_instruction_mode(ABSOLUTE_DISTANCE_MODES)
        tpl = f"size.{stem_kind}.single_view.{mode}"
        x_val = format_distance_value(value_m)
        prompt = self.render_structured_prompt(
            tpl, shared={"A": A_desc, "X": x_val},
        )
        return prompt, tpl

    def _generate_absolute_size(self, graph):
        if not graph.is_metric_depth:
            return None
        nodes = [n for n in graph.get_object_nodes() if n.box_3d_world is not None]
        if len(nodes) == 0:
            return None
        num = random.randint(2, min(len(nodes), 4)) if len(nodes) > 1 else 1
        sampled = random.sample(nodes, num)
        qa_image = graph.primary_view.image if self.emit_marked_images else None
        processed_image, marked = self.mark_objects_for_qa(qa_image, sampled)
        mark_spec = self.marker.last_mark_spec

        prompts = []
        for m in marked:
            p, tpl = self.absolute_size_prompt_func(
                m, "absolute",
                lambda n: max(self._get_node_extent(n)),
            )
            prompts.append(p)
            self._record_turn(
                "absolute_size", tpl, p, QuestionType.OPEN_ENDED,
                mark_spec=mark_spec, extra_slots=self._slots_from_marked([m]),
            )
            p2, tpl2 = self.absolute_size_prompt_func(
                m, "height",
                lambda n: n.box_3d_world[5],
            )
            prompts.append(p2)
            self._record_turn(
                "absolute_size", tpl2, p2, QuestionType.OPEN_ENDED,
                mark_spec=mark_spec, extra_slots=self._slots_from_marked([m]),
            )

        return [
            (p, processed_image, QuestionType.OPEN_ENDED) for p in prompts
        ]

    def _generate_relative_size(self, graph):
        nodes = [n for n in graph.get_object_nodes() if n.box_3d_world is not None]
        if len(nodes) < 2:
            return None
        sampled = random.sample(nodes, 2)
        qa_image = graph.primary_view.image if self.emit_marked_images else None
        processed_image, marked = self.mark_objects_for_qa(qa_image, sampled)
        prompt, tpl = self.relative_size_prompt_func(marked[0], marked[1])
        if prompt is None:
            return None
        self._record_turn(
            "relative_size", tpl, prompt, QuestionType.JUDGMENT,
            mark_spec=self.marker.last_mark_spec,
            extra_slots=self._slots_from_marked(marked),
        )
        return prompt, processed_image, QuestionType.JUDGMENT
