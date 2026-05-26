"""
Position annotation task: height comparison & near/far (proximity).

Templates: position.height_higher | height_lower | near_far (MCQ: direct / sentence / free)
"""

import random
import numpy as np
from itertools import combinations
from .core.base_annotation_task import BaseAnnotationTask
from .core.sample_metadata import marked_surface_label
from .core.visual_marker import MarkConfig
from utils.box_utils import compute_box_3d_corners
from .core.question_type import QuestionType

from utils.point_cloud_utils import compute_point_cloud_distance
from utils.image_utils import convert_pil_to_bytes


class AnnotationGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Singleview Position"
    SUB_TASKS = {
        "height_comparison": {"default": 1, "handler": "_generate_height_comparison"},
        "proximity":         {"default": 1, "handler": "_generate_proximity"},
    }

    _NEAR_LABEL = "near each other"
    _FAR_LABEL = "far from each other"

    def __init__(self, args):
        super().__init__(args)
        self.task_name = args.get("task_name") or args.get("file_name", "position")

    def get_mark_config(self):
        return MarkConfig(mark_types=["box", "point"])

    @staticmethod
    def _format_mcq_answer(letter: str, option_text: str) -> str:
        return f"{letter}. {option_text}"

    def _get_z_max(self, node):
        box_3d_world = node.box_3d_world
        corners = compute_box_3d_corners(np.array(box_3d_world[:3]), np.array(box_3d_world[3:6]), box_3d_world[6:])
        return np.max(corners[:, -1])

    def height_comparison_prompt_func(self, A, B):
        A_desc = marked_surface_label(A)
        B_desc = marked_surface_label(B)
        _, A_node = A
        _, B_node = B

        is_above = self._get_z_max(A_node) > self._get_z_max(B_node)

        if random.random() < 0.5:
            base_tpl = "position.height_higher"
            target = A_desc if is_above else B_desc
        else:
            base_tpl = "position.height_lower"
            target = A_desc if not is_above else B_desc

        opt_a = f"The {A_desc}"
        opt_b = f"The {B_desc}"
        options = f"\nOptions: A. {opt_a} B. {opt_b}"
        letter = "A" if target == A_desc else "B"
        option_text = opt_a if letter == "A" else opt_b
        answer = self._format_mcq_answer(letter, option_text)

        higher_desc = A_desc if is_above else B_desc
        lower_desc = B_desc if is_above else A_desc
        if base_tpl == "position.height_higher":
            premise = f"The {higher_desc} is at a higher elevation than the {lower_desc}"
        else:
            premise = f"The {lower_desc} is at a lower elevation than the {higher_desc}"

        prompt = self.render_structured_prompt(
            base_tpl,
            shared={
                "P": premise,
                "A": A_desc,
                "B": B_desc,
                "O": options,
                "X": answer,
            },
        )
        return prompt, base_tpl

    def proximity_prompt_func(self, A, B, near_or_far):
        A_desc, B_desc = marked_surface_label(A), marked_surface_label(B)

        labels = [self._NEAR_LABEL, self._FAR_LABEL]
        if random.random() < 0.5:
            labels = labels[::-1]

        options = f"\nOptions: A. {labels[0]} B. {labels[1]}"
        tpl_name = "position.near_far"

        if near_or_far == "near":
            letter = "A" if labels[0] == self._NEAR_LABEL else "B"
        else:
            letter = "A" if labels[0] == self._FAR_LABEL else "B"
        option_text = labels[0] if letter == "A" else labels[1]
        answer = self._format_mcq_answer(letter, option_text)

        if near_or_far == "near":
            premise = f"The {A_desc} and the {B_desc} are near each other"
        else:
            premise = f"The {A_desc} and the {B_desc} are far from each other"

        prompt = self.render_structured_prompt(
            tpl_name,
            shared={
                "P": premise,
                "A": A_desc,
                "B": B_desc,
                "O": options,
                "X": answer,
            },
        )
        return prompt, tpl_name

    def _generate_height_comparison(self, graph):
        nodes = [n for n in graph.get_object_nodes() if n.box_3d_world is not None]
        if len(nodes) < 2:
            return None
        image = graph.primary_view.image
        sampled = random.sample(nodes, 2)
        qtype = QuestionType.MCQ

        if random.random() < 0.8:
            processed_image, marked = self.mark_objects_for_qa(image, sampled)
        else:
            processed_image = {"bytes": convert_pil_to_bytes(image)}
            marked = [(n.tag, n) for n in sampled]

        prompt, tpl = self.height_comparison_prompt_func(marked[0], marked[1])
        self._record_turn(
            "height_comparison", tpl, prompt, qtype,
            mark_spec=self.marker.last_mark_spec,
            extra_slots=self._slots_from_marked(marked),
        )
        return prompt, processed_image, qtype

    def _generate_proximity(self, graph):
        nodes = [n for n in graph.get_object_nodes() if n.box_3d_world is not None]
        if len(nodes) < 2:
            return None
        if len(nodes) > 8:
            nodes = random.sample(nodes, 8)
        image = graph.primary_view.image

        near_candidates, far_candidates = [], []
        for nodeA, nodeB in combinations(nodes, 2):
            if nodeA.tag == nodeB.tag:
                continue
            pcA = nodeA.view_appearances[0].pointcloud_camera
            pcB = nodeB.view_appearances[0].pointcloud_camera
            distance = compute_point_cloud_distance(pcA, pcB)
            avg_extent = np.mean([
                np.mean(nodeA.box_3d_world[3:6]),
                np.mean(nodeB.box_3d_world[3:6]),
            ])
            ratio = distance / avg_extent if avg_extent > 0 else float('inf')
            if ratio < 0.5:
                near_candidates.append((nodeA, nodeB))
            elif ratio > 2.0:
                far_candidates.append((nodeA, nodeB))

        if not near_candidates and not far_candidates:
            return None

        results = []
        if near_candidates:
            pair = random.choice(near_candidates)
            if random.random() < 0.5:
                processed_image, marked = self.mark_objects_for_qa(image, list(pair))
            else:
                processed_image = {"bytes": convert_pil_to_bytes(image)}
                marked = [(n.tag, n) for n in pair]
            prompt, tpl = self.proximity_prompt_func(marked[0], marked[1], "near")
            self._record_turn(
                "proximity", tpl, prompt, QuestionType.MCQ,
                mark_spec=self.marker.last_mark_spec,
                extra_slots=self._slots_from_marked(marked),
            )
            results.append((prompt, processed_image, QuestionType.MCQ))
        if far_candidates:
            pair = random.choice(far_candidates)
            if random.random() < 0.5:
                processed_image, marked = self.mark_objects_for_qa(image, list(pair))
            else:
                processed_image = {"bytes": convert_pil_to_bytes(image)}
                marked = [(n.tag, n) for n in pair]
            prompt, tpl = self.proximity_prompt_func(marked[0], marked[1], "far")
            self._record_turn(
                "proximity", tpl, prompt, QuestionType.MCQ,
                mark_spec=self.marker.last_mark_spec,
                extra_slots=self._slots_from_marked(marked),
            )
            results.append((prompt, processed_image, QuestionType.MCQ))

        return results
