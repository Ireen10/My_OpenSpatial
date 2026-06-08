"""
Multiview distance: pair absolute (2 views) + N-ary farthest/closest (3�? views).

Templates:
    multiview_distance.absolute.{direct,sentence,free}  (stems shared with singleview; + introduction)
    multiview_distance.{farthest,closest}.{direct,reasoning,free}
"""

from task.annotation.core.thread_rng import rng
from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType

from utils.point_cloud_utils import compute_point_cloud_distance

from .metric_gating import (
    ABSOLUTE_DISTANCE_MODES,
    format_distance_value,
    pick_instruction_mode,
    pick_metric_gated_mode,
)


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Multiview Distance"
    SUB_TASKS = {
        "pair_absolute_distance":  {"default": 1, "handler": "_generate_pair_absolute_distance"},
        "multi_relative_distance": {"default": 1, "handler": "_generate_multi_relative_distance"},
    }
    _INSTRUCTION_MODES = ("direct", "reasoning", "free")

    def _pick_instruction_mode(self, graph) -> str:
        """Reasoning/free question instructions require metric depth."""
        return pick_metric_gated_mode(
            self._INSTRUCTION_MODES,
            is_metric_depth=graph.is_metric_depth,
            metric_only_modes=("reasoning", "free"),
        )

    def get_mark_config(self):
        return MarkConfig(type_weights={"point": 0.3, "box": 0.7})

    # ─── Prompt Functions ─────────────────────────────────────────────

    def pair_absolute_distance_prompt_func(self, A, B):
        """Absolute distance QA (same stems/answers as singleview; multiview template adds introduction)."""
        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_cloud = self._clean_cloud(A_cloud)
        B_cloud = self._clean_cloud(B_cloud)

        dist_m = compute_point_cloud_distance(A_cloud, B_cloud)
        mode = pick_instruction_mode(ABSOLUTE_DISTANCE_MODES)
        tpl = f"multiview_distance.absolute.{mode}"
        x_val = format_distance_value(dist_m, scaling_factor=self.scaling_factor)

        prompt = self.render_structured_prompt(
            tpl,
            shared={"A": A_desc, "B": B_desc, "X": x_val},
        )
        return prompt, tpl

    def multi_relative_distance_prompt_func(self, obj_infos, graph):
        """N-ary farthest/closest to a reference object across multi-view marks."""
        distance_type = rng().choice(["closest", "farthest"])
        mode = self._pick_instruction_mode(graph)
        tpl = f"multiview_distance.{distance_type}.{mode}"

        ref_idx = rng().randint(0, len(obj_infos) - 1)
        ref_desc, ref_cloud = obj_infos[ref_idx]
        ref_desc = ref_desc.lower()
        ref_cloud = self._clean_cloud(ref_cloud)

        candidates = [obj_infos[i] for i in range(len(obj_infos)) if i != ref_idx]
        distances = []
        for cand_desc, cand_cloud in candidates:
            cand_cloud = self._clean_cloud(cand_cloud)
            dist = compute_point_cloud_distance(ref_cloud, cand_cloud)
            distances.append((dist, cand_desc.lower()))

        distances.sort(key=lambda x: x[0], reverse=(distance_type == "farthest"))
        target_desc = distances[0][1]
        all_descs = [d for _, d in distances]
        rng().shuffle(all_descs)

        prompt = self.render_structured_prompt(
            tpl,
            shared={"T": ", ".join(all_descs)},
            q_args={"X": ref_desc},
            a_args={"X": target_desc, "Y": ref_desc},
        )
        return prompt, tpl

    # ─── Handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _generate_pair_absolute_distance(self, graph):
        if not graph.is_metric_depth:
            return None
        result = self._find_chain_and_mark(graph, num_views=2)
        if result is None:
            return None
        meta, processed_images, objs = result
        prompt_items = self._marked_prompt_items(meta, objs)
        prompt, tpl = self.pair_absolute_distance_prompt_func(
            prompt_items[0], prompt_items[1],
        )
        self._record_multiview_turn(
            "pair_absolute_distance", tpl, prompt, QuestionType.OPEN_ENDED, meta,
            mark_spec=self.marker.last_mark_spec,
            extra_slots=self._slots_from_marked(objs, labels=["A", "B"]),
        )
        return prompt, processed_images, QuestionType.OPEN_ENDED

    def _generate_multi_relative_distance(self, graph):
        result = self._find_chain_and_mark(graph, num_views=rng().choice([3, 4, 5, 6]))
        if result is None:
            return None
        meta, processed_images, objs = result
        prompt, tpl = self.multi_relative_distance_prompt_func(
            self._marked_prompt_items(meta, objs), graph,
        )
        labels = [chr(ord("A") + i) for i in range(len(objs))]
        self._record_multiview_turn(
            "multi_relative_distance", tpl, prompt, QuestionType.OPEN_ENDED, meta,
            mark_spec=self.marker.last_mark_spec,
            extra_slots=self._slots_from_marked(objs, labels=labels),
        )
        return prompt, processed_images, QuestionType.OPEN_ENDED
