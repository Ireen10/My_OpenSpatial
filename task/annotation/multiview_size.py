"""
Multiview size: pair relative judgment (2 views) + N-ary biggest/smallest (3–6 views).

Templates:
    multiview_size.{big,small}.pair  — Judgment (D_diag + 1.2 ratio gate, aligned with singleview)
    multiview_size.{biggest,smallest}.{direct,sentence}  — N-ary (D_diag; 1.2 gate on rank-1 vs rank-2 only)
"""

from task.annotation.core.thread_rng import rng
from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType
from utils.box_utils import RELATIVE_SIZE_DIAG_RATIO_MIN, box_3d_diag_extent

from .metric_gating import pick_instruction_mode


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Multiview Size"
    SUB_TASKS = {
        "pair_relative_size":  {"default": 1, "handler": "_generate_pair_relative_size"},
        "multi_relative_size": {"default": 1, "handler": "_generate_multi_relative_size"},
    }
    _SUPERLATIVE_MODES = ("direct", "sentence", "free")

    def get_mark_config(self):
        return MarkConfig(mark_types=["box", "point"])

    # ─── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _diag_from_box_or_cloud(cloud, box_3d_world=None):
        """N-ary superlative: prefer 3D box, else point-cloud AABB extent."""
        if box_3d_world is not None:
            return box_3d_diag_extent(box_3d_world)
        ext = cloud.get_axis_aligned_bounding_box().get_extent()
        return box_3d_diag_extent([0, 0, 0, ext[0], ext[1], ext[2]])

    # ─── Prompt Functions ─────────────────────────────────────────────

    def pair_relative_size_prompt_func(self, A, B, boxes_3d_world=None):
        """Pair judgment: same metric as singleview relative_size (box D_diag + ratio gate)."""
        A_desc, _ = A
        B_desc, _ = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        if (
            boxes_3d_world is None
            or boxes_3d_world[0] is None
            or boxes_3d_world[1] is None
        ):
            return None, None
        d_A = box_3d_diag_extent(boxes_3d_world[0])
        d_B = box_3d_diag_extent(boxes_3d_world[1])
        lo, hi = min(d_A, d_B), max(d_A, d_B)
        if lo <= 0 or hi / lo < RELATIVE_SIZE_DIAG_RATIO_MIN:
            return None, None

        tpl_name = rng().choice(["multiview_size.big.pair", "multiview_size.small.pair"])
        is_bigger = d_A > d_B
        condition = is_bigger if "big" in tpl_name else not is_bigger

        if not self.register_semantic_candidate(
            "multiview_size.pair", tpl_name,
            sorted([str(A_desc), str(B_desc)]), bool(condition),
        ):
            return None, None
        prompt = self.render_structured_prompt(tpl_name, condition=condition, shared={"A": A_desc, "B": B_desc})
        return prompt, tpl_name

    def multi_relative_size_prompt_func(self, obj_infos, boxes_3d_world=None):
        """Generate a superlative size QA for N objects from different views."""
        size_type = rng().choice(["biggest", "smallest"])
        diags = []
        for i, (desc, cloud) in enumerate(obj_infos):
            d = self._diag_from_box_or_cloud(
                cloud, boxes_3d_world[i] if boxes_3d_world else None,
            )
            diags.append((d, desc.lower()))

        diags.sort(key=lambda x: x[0], reverse=(size_type == "biggest"))
        if not diags or diags[0][0] <= 0:
            return None, None
        if len(diags) >= 2:
            top, second = diags[0][0], diags[1][0]
            if second <= 0 or top / second < RELATIVE_SIZE_DIAG_RATIO_MIN:
                return None, None
        target_desc = diags[0][1]
        all_tags = [d for _, d in diags]
        rng().shuffle(all_tags)

        if not self.register_semantic_candidate(
            "multiview_size.superlative", size_type,
            sorted(all_tags), target_desc,
        ):
            return None, None
        mode = pick_instruction_mode(self._SUPERLATIVE_MODES)
        tpl_name = f"multiview_size.{size_type}.{mode}"

        prompt = self.render_structured_prompt(
            tpl_name, shared={"T": ", ".join(all_tags), "X": target_desc},
        )
        return prompt, tpl_name

    # ─── Handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _generate_pair_relative_size(self, graph):
        result = self._find_chain_and_mark(graph, num_views=2)
        if result is None:
            return None
        meta, processed_images, objs = result
        prompt_items = self._marked_prompt_items(meta, objs)
        prompt, tpl = self.pair_relative_size_prompt_func(
            prompt_items[0], prompt_items[1], boxes_3d_world=meta["box_3d_world"],
        )
        if prompt is None:
            return None
        self._record_multiview_turn(
            "pair_relative_size", tpl, prompt, QuestionType.JUDGMENT, meta,
            mark_spec=self.marker.last_mark_spec,
            extra_slots=self._slots_from_marked(objs, labels=["A", "B"]),
        )
        return prompt, processed_images, QuestionType.JUDGMENT

    def _generate_multi_relative_size(self, graph):
        result = self._find_chain_and_mark(graph, num_views=rng().choice([3, 4, 5, 6]))
        if result is None:
            return None
        meta, processed_images, objs = result
        prompt, tpl = self.multi_relative_size_prompt_func(
            self._marked_prompt_items(meta, objs), boxes_3d_world=meta["box_3d_world"],
        )
        if prompt is None:
            return None
        labels = [chr(ord("A") + i) for i in range(len(objs))]
        self._record_multiview_turn(
            "multi_relative_size", tpl, prompt, QuestionType.OPEN_ENDED, meta,
            mark_spec=self.marker.last_mark_spec,
            extra_slots=self._slots_from_marked(objs, labels=labels),
        )
        return prompt, processed_images, QuestionType.OPEN_ENDED
