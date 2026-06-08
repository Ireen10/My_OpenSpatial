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
    distance.relative_{far,close}_mcq.{direct,reasoning,free}
      (reasoning/free answers use is_metric_depth for metric vs semantic templates)

Mark policy (see MARK_TASKS_MEMO.md): ambiguous tag in view → mark_spec required;
else 25% optional mark_spec. QA images stay unmarked unless emit_marked_images.
"""

from task.annotation.core.thread_rng import rng

from .core.base_annotation_task import BaseAnnotationTask
from .core.sample_metadata import marked_surface_label
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType

from utils.point_cloud_utils import compute_point_cloud_distance

from .metric_gating import (
    ABSOLUTE_DISTANCE_MODES,
    format_distance_value,
    pick_instruction_mode,
    pick_relative_distance_mode,
    pick_stem_index,
)


class AnnotationGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Singleview Distance"
    SUB_TASKS = {
        "absolute_distance": {"default": 1, "handler": "_generate_absolute_distance"},
        "relative_distance": {"default": 1, "handler": "_generate_relative_distance"},
    }


    def __init__(self, args):
        super().__init__(args)
        self.task_name = args.get("task_name") or args.get("file_name", "distance")

    def get_mark_config(self):
        return MarkConfig(type_weights={"point": 0.25, "box": 0.75})

    @staticmethod
    def _title_desc(desc: str) -> str:
        desc = (desc or "").strip()
        if not desc:
            return desc
        return desc[0].upper() + desc[1:]

    @staticmethod
    def _fmt_dist_m(dist: float) -> str:
        return f"{dist:.2f} m"


    def _relative_stem_index(self, template_id: str, graph) -> int:
        tpl = self.get_structured_template(template_id)
        return pick_stem_index(tpl.stem, is_metric_depth=graph.is_metric_depth)

    def _get_cleaned_cloud(self, marked):
        """Extract and clean pointcloud from a marked result (desc, node)."""
        _, cloud = self._get_cloud(marked)
        return marked_surface_label(marked), self._clean_cloud(cloud)

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

        dist_m = compute_point_cloud_distance(A_cloud, B_cloud)
        mode = pick_instruction_mode(ABSOLUTE_DISTANCE_MODES)
        tpl = f"distance.absolute.{mode}"
        x_val = format_distance_value(dist_m, scaling_factor=self.scaling_factor)

        prompt = self.render_structured_prompt(
            tpl,
            shared={"A": A_desc, "B": B_desc, "X": x_val},
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

        if rng().random() < 0.5:
            polarity, winner, other = "far", farther, closer
        else:
            polarity, winner, other = "close", closer, farther

        mode = pick_relative_distance_mode(is_metric_depth=graph.is_metric_depth)
        tpl = f"distance.relative_{polarity}.{mode}"
        stem_index = self._relative_stem_index(tpl, graph)
        shared = {
            "A": A_desc,
            "B": B_desc,
            "C": C_desc,
            "D": self._fmt_dist_m(dist_AC),
            "E": self._fmt_dist_m(dist_BC),
            "O": "",
            "X": self._title_desc(winner),
            "G": self._title_desc(other),
        }

        prompt = self.render_structured_prompt(
            tpl,
            shared=shared,
            is_metric_depth=graph.is_metric_depth,
            stem_index=stem_index,
        )
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

        if rng().random() < 0.5:
            polarity, target_tag, winner_desc = "far", farther_tag, farther
        else:
            polarity, target_tag, winner_desc = "close", closer_tag, closer

        mode = pick_relative_distance_mode(is_metric_depth=graph.is_metric_depth)
        tpl = f"distance.relative_{polarity}_mcq.{mode}"
        stem_index = self._relative_stem_index(tpl, graph)
        option_answer = self._mcq_option_label(
            target_tag, A_desc, B_desc,
        )
        other_desc = B_desc if target_tag == "A" else A_desc
        anchor = self._title_desc(C_desc)
        if polarity == "far":
            premise = f"{self._title_desc(winner_desc)} is farther from the {anchor}"
        else:
            premise = f"{self._title_desc(winner_desc)} is closer to the {anchor}"
        shared = {
            "A": A_desc,
            "B": B_desc,
            "C": C_desc,
            "D": self._fmt_dist_m(dist_AC),
            "E": self._fmt_dist_m(dist_BC),
            "F": self._title_desc(winner_desc),
            "G": self._title_desc(other_desc),
            "P": premise,
            "O": options,
            "X": option_answer,
        }

        prompt = self.render_structured_prompt(
            tpl,
            shared=shared,
            is_metric_depth=graph.is_metric_depth,
            stem_index=stem_index,
        )
        return prompt, tpl

    def _generate_absolute_distance(self, graph):
        """Measure the absolute distance between two objects."""
        if not graph.is_metric_depth:
            return None
        nodes = graph.get_object_nodes()
        if len(nodes) < 2:
            return None
        image = graph.primary_view.image
        sampled = rng().sample(nodes, 2)
        processed_image, marked = self.mark_objects_for_qa(image, sampled)
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
        sampled = rng().sample(nodes, 3)
        processed_image, marked = self.mark_objects_for_qa(image, sampled)
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
