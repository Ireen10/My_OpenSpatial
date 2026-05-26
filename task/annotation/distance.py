"""
Distance annotation task: absolute distance & relative distance comparison.

Sub-tasks:
    absolute_distance — measure the point-cloud minimum distance between two objects
                        and report in metres or centimetres (requires metric depth).
    relative_distance — given three objects A, B, C, ask which of A/B is farther from /
                        closer to C, in both open-ended and MCQ form.

Templates used (singleview):
    distance.absolute_{m,cm}.{direct,sentence}
    distance.relative_{far,close}.{direct,reasoning,free}
      (reasoning/free require graph.is_metric_depth; else fallback to direct)
    distance.relative_{far,close}_mcq.{direct,reasoning,free}
"""

import random

from .core.base_annotation_task import BaseAnnotationTask
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType

from utils.point_cloud_utils import compute_point_cloud_distance


class AnnotationGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Singleview Distance"
    SUB_TASKS = {
        "absolute_distance": {"default": 1, "handler": "_generate_absolute_distance"},
        "relative_distance": {"default": 1, "handler": "_generate_relative_distance"},
    }

    # free: question_instruction pool empty (project-wide M8 convention)
    _RELATIVE_MODES = ("direct", "reasoning", "free")

    def __init__(self, args):
        super().__init__(args)
        self.task_name = args.get("task_name") or args.get("file_name", "distance")

    def get_mark_config(self):
        return MarkConfig(type_weights={"mask": 0.2, "box": 0.8})

    @staticmethod
    def _title_desc(desc: str) -> str:
        desc = (desc or "").strip()
        if not desc:
            return desc
        return desc[0].upper() + desc[1:]

    @staticmethod
    def _fmt_dist_m(dist: float) -> str:
        return f"{dist:.2f} m"

    def _pick_relative_mode(self, graph) -> str:
        """Sample instruction mode; reasoning/free need metric depth, else use direct."""
        mode = random.choice(self._RELATIVE_MODES)
        if mode in ("reasoning", "free") and not graph.is_metric_depth:
            return "direct"
        return mode

    def _get_cleaned_cloud(self, marked):
        """Extract and clean pointcloud from a marked result (desc, node)."""
        desc, cloud = self._get_cloud(marked)
        return desc.lower(), self._clean_cloud(cloud)

    def _resolve_relative_distance(self, A, B, C):
        """Return descriptors, winner text/tags, and anchor distances (metres)."""
        A_desc, A_cloud = self._get_cleaned_cloud(A)
        B_desc, B_cloud = self._get_cleaned_cloud(B)
        C_desc, C_cloud = self._get_cleaned_cloud(C)

        dist_AC = compute_point_cloud_distance(A_cloud, C_cloud)
        dist_BC = compute_point_cloud_distance(B_cloud, C_cloud)

        if dist_AC > dist_BC:
            farther, closer = A_desc, B_desc
            farther_tag, closer_tag = "A", "B"
        else:
            farther, closer = B_desc, A_desc
            farther_tag, closer_tag = "B", "A"

        return (
            A_desc,
            B_desc,
            C_desc,
            farther,
            closer,
            farther_tag,
            closer_tag,
            dist_AC,
            dist_BC,
        )

    @staticmethod
    def _mcq_option_label(tag: str, a_desc: str, b_desc: str) -> str:
        desc = a_desc if tag == "A" else b_desc
        return f"{tag}. {AnnotationGenerator._title_desc(desc)}"

    def absolute_distance_prompt_func(self, A, B):
        """Generate an absolute distance QA for two marked objects."""
        A_desc, A_cloud = self._get_cleaned_cloud(A)
        B_desc, B_cloud = self._get_cleaned_cloud(B)

        unit = random.choice(["m", "cm"])
        dist = compute_point_cloud_distance(A_cloud, B_cloud)
        scaled = dist * self.scaling_factor * (100 if unit == "cm" else 1)
        style = random.choice(["direct", "sentence"])
        tpl = f"distance.absolute_{unit}.{style}"

        prompt = self.render_structured_prompt(
            tpl,
            shared={"A": A_desc, "B": B_desc, "X": f"{scaled:.2f}"},
        )
        return prompt, tpl

    def relative_distance_oe_prompt_func(self, A, B, C, *, graph):
        """Open-ended relative distance with instruction-constrained answer forms."""
        (
            A_desc,
            B_desc,
            C_desc,
            farther,
            closer,
            _,
            _,
            dist_AC,
            dist_BC,
        ) = self._resolve_relative_distance(A, B, C)

        if random.random() < 0.5:
            polarity, winner = "far", farther
        else:
            polarity, winner = "close", closer

        mode = self._pick_relative_mode(graph)
        tpl = f"distance.relative_{polarity}.{mode}"
        shared = {
            "A": A_desc,
            "B": B_desc,
            "C": C_desc,
            "D": self._fmt_dist_m(dist_AC),
            "E": self._fmt_dist_m(dist_BC),
            "O": "",
            "X": self._title_desc(winner),
        }

        prompt = self.render_structured_prompt(tpl, shared=shared)
        return prompt, tpl

    def relative_distance_mcq_prompt_func(self, A, B, C, *, graph):
        """MCQ relative distance with instruction-constrained answer forms."""
        (
            A_desc,
            B_desc,
            C_desc,
            farther,
            closer,
            farther_tag,
            closer_tag,
            dist_AC,
            dist_BC,
        ) = self._resolve_relative_distance(A, B, C)

        options = f"\nOptions: A. {self._title_desc(A_desc)}  B. {self._title_desc(B_desc)}."

        if random.random() < 0.5:
            polarity, target_tag = "far", farther_tag
        else:
            polarity, target_tag = "close", closer_tag

        mode = self._pick_relative_mode(graph)
        tpl = f"distance.relative_{polarity}_mcq.{mode}"
        option_answer = self._mcq_option_label(
            target_tag, A_desc, B_desc,
        )
        shared = {
            "A": A_desc,
            "B": B_desc,
            "C": C_desc,
            "D": self._fmt_dist_m(dist_AC),
            "E": self._fmt_dist_m(dist_BC),
            "O": options,
            "X": option_answer,
        }

        prompt = self.render_structured_prompt(tpl, shared=shared)
        return prompt, tpl

    def _generate_absolute_distance(self, graph):
        """Measure the absolute distance between two objects."""
        if not graph.is_metric_depth:
            return None
        nodes = graph.get_object_nodes()
        if len(nodes) < 2:
            return None
        image = graph.primary_view.image
        sampled = random.sample(nodes, 2)
        processed_image, marked = self.mark_objects_for_qa(image, sampled, mark_prob=1.0)
        A, B = marked
        prompt, tpl = self.absolute_distance_prompt_func(A, B)
        self._record_turn(
            "absolute_distance",
            tpl,
            prompt,
            QuestionType.OPEN_ENDED,
            mark_spec=self.marker.last_mark_spec,
            extra_slots=self._slots_from_marked(marked),
        )
        return prompt, processed_image, QuestionType.OPEN_ENDED

    def _generate_relative_distance(self, graph):
        """Sample three objects and ask relative distance to the third."""
        nodes = graph.get_object_nodes()
        if len(graph.obj_tags) <= 2:
            return None
        image = graph.primary_view.image
        sampled = random.sample(nodes, 3)
        processed_image, marked = self.mark_objects_for_qa(image, sampled, mark_prob=1.0)
        A, B, C = marked
        slots = self._slots_from_marked(marked, labels=["A", "B", "C"])
        mark_spec = self.marker.last_mark_spec

        oe_prompt, oe_tpl = self.relative_distance_oe_prompt_func(A, B, C, graph=graph)
        self._record_turn(
            "relative_distance",
            oe_tpl,
            oe_prompt,
            QuestionType.OPEN_ENDED,
            mark_spec=mark_spec,
            extra_slots=slots,
        )
        mcq_prompt, mcq_tpl = self.relative_distance_mcq_prompt_func(A, B, C, graph=graph)
        self._record_turn(
            "relative_distance",
            mcq_tpl,
            mcq_prompt,
            QuestionType.MCQ,
            mark_spec=mark_spec,
            extra_slots=slots,
        )

        return [
            (oe_prompt, processed_image, QuestionType.OPEN_ENDED),
            (mcq_prompt, processed_image, QuestionType.MCQ),
        ]
