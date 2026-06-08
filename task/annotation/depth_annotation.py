"""
Depth estimation annotation task: depth ordering & depth choice.

Sub-tasks:
    depth_ordering_oe  �?sort 3-5 marked objects/points by depth (near→far), open-ended.
    depth_ordering_mcq �?same ordering task but as 4-option MCQ (exactly 4 points/objects).
    depth_choice_oe    �?pick the closest / farthest / N-th closest object, open-ended.
    depth_choice_mcq   �?same choice task but as 4-option MCQ.

Visual annotation modes:
    random_sample (~10%) �?sample 4-7 random pixels with distinct depths; returns
                           normalized [x, y] coordinate tags. No drawing on image.
    object-based  (~90%) �?select 3-5 nodes, plan point/box mark_spec,
                           then compute per-object depth from the depth_map.

Depth estimation:
    For each object, depth = mean of the shallowest 10% of valid mask pixels.
    This approximates the front-surface depth and is robust to noisy backgrounds.

Templates used:
    depth.ordering       �?[T] type label, [A] object list, [X] sorted list
    depth.ordering_mcq   �?[T] type label, [Y] options; [X] obj list (q) / answer (a)
    depth.farthest       �?[T] type label, [A] object list, [X] answer
    depth.closest        �?[T] type label, [A] object list, [X] answer
    depth.choice         �?[T] type label, [A] object list, [B] ordinal, [X] answer
    depth.farthest_mcq   �?[T] type label, [Y] options; [X] obj list (q) / answer (a)
    depth.closest_mcq    �?[T] type label, [Y] options; [X] obj list (q) / answer (a)
    depth.choice_mcq     �?[T] type label, [Y] ordinal, [Z] options; [X] obj list (q) / answer (a)
"""

from task.annotation.core.thread_rng import rng
import numpy as np
from .core.base_annotation_task import BaseAnnotationTask
from .core.mark_spec import plan_point_marks
from .core.visual_marker import MarkConfig, VisualMarker
from .core.question_type import QuestionType
from .core.mark_spec import render_mark

from .metric_gating import pick_instruction_mode

_DEPTH_INSTRUCTION_MODES = ("direct", "sentence", "free")

ORDINALS = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth",
]

TASK_NAME = "depth_annotation"


class AnnotationGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Singleview Depth Estimation"
    SUB_TASKS = {
        "depth_ordering_oe":  {"default": 1, "handler": "_generate_depth_ordering_oe"},
        "depth_ordering_mcq": {"default": 1, "handler": "_generate_depth_ordering_mcq"},
        "depth_choice_oe":    {"default": 1, "handler": "_generate_depth_choice_oe"},
        "depth_choice_mcq":   {"default": 1, "handler": "_generate_depth_choice_mcq"},
    }

    # ── config ────────────────────────────────────────────────────────

    def __init__(self, args):
        super().__init__(args)
        self.task_name = TASK_NAME
        self.emit_metadata = bool(args.get("emit_metadata", True))
        self.emit_marked_images = bool(args.get("emit_marked_images", False))

    def get_mark_config(self):
        """50% point / 50% box (mask marks disabled pipeline-wide)."""
        return MarkConfig(type_weights={"point": 0.5, "box": 0.5})

    def check_example(self, example) -> bool:
        """Require image, obj_tags, depth_map, masks, and >= 3 objects."""
        if not super().check_example(example):
            return False
        if "depth_map" not in example:
            return False
        if "masks" not in example:
            return False
        return len(example["obj_tags"]) >= 3

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_depth(mask, depth_map):
        """Estimate front-surface depth from a binary mask.

        Takes the shallowest 10% of valid (>0) depth pixels inside the mask
        and returns their mean. Falls back to the full mean when the mask
        covers very few pixels (k >= total count).
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        depths = depth_map[ys, xs]
        depths = depths[depths > 0]
        if len(depths) == 0:
            return None
        k = max(1, int(len(depths) * 0.1))
        if k >= len(depths):
            return float(np.mean(depths))
        return float(np.mean(np.partition(depths, k)[:k]))

    # ── data preparation ─────────────────────────────────────────────

    def _sample_random_points(self, depth_map, *, num_points=None):
        """Generate coordinate-based tags by sampling random pixels.

        Picks 4-7 pixels (or exactly ``num_points`` when set) whose depths differ
        by > 0.05 from all previously selected pixels. Coordinates are normalized
        to a [0, 1000] grid. The original image is returned unchanged (no drawing).

        Returns (tags, depth_sorted_tags, image_bytes, mark_spec) or None on failure.
        """
        h, w = depth_map.shape
        if num_points is None:
            num_points = rng().randint(4, 7)
        points, selected_depths = [], []
        for _ in range(num_points):
            for _ in range(100):
                u, v = rng().randint(0, w - 1), rng().randint(0, h - 1)
                d = depth_map[v, u]
                if d > 0 and all(abs(d - sd) > 0.05 for sd in selected_depths):
                    points.append([u, v])
                    selected_depths.append(d)
                    break
        if len(points) != num_points:
            return None
        sorted_pts = [points[i] for i in np.argsort(selected_depths)]
        norm = lambda p: str([int(p[0] / w * 1000), int(p[1] / h * 1000)])
        self.marker.reset(shuffle=True)
        mark_spec, _ = plan_point_marks(self.marker, points)
        return (
            [norm(p) for p in points],
            [norm(p) for p in sorted_pts],
            None,
            mark_spec,
        )

    def _mark_and_sort(self, image, depth_map, nodes, *, num_objects=None):
        """Plan mark_spec, optionally render, sort objects by depth.

        Returns (legacy_tags, semantic_tags, depth_sorted_semantic, image_bytes, mark_spec)
        or None.
        """
        if num_objects is None:
            num = rng().randint(3, min(5, len(nodes)))
        else:
            num = min(num_objects, len(nodes))
        sampled = rng().sample(nodes, num)

        graph = getattr(self._thread_local, "scene_graph", None)
        enable = self.resolve_mark_enabled(graph, sampled, view_idx=0)

        self.marker.reset(shuffle=True)
        mark_type = self.marker.choose_mark_type()
        if enable:
            spec, marked_info = self.marker.plan_mark(sampled, mark_type=mark_type)
        else:
            spec = {"version": 2, "mark_kinds": [], "slots": []}
            marked_info = [(n.tag, n) for n in sampled]
        self.marker._last_mark_spec = spec if enable else None
        preprocess_row = getattr(self._thread_local, "preprocess_row", None)
        if self.emit_marked_images and enable:
            processed_image = render_mark(image, spec, preprocess_row=preprocess_row)
        else:
            processed_image = None

        tags_legacy, tags_sem, depths = [], [], []
        kept_slots = []

        def _append_depth(node, legacy_label: str, semantic_label: str, slot=None):
            app = node.view_appearances.get(0)
            if app is None:
                return
            mask = np.array(app.mask)
            depth = self._compute_depth(mask, depth_map)
            if depth is None:
                return
            tags_legacy.append(legacy_label)
            tags_sem.append(semantic_label)
            depths.append(depth)
            if slot is not None:
                kept_slots.append(slot)

        if enable:
            for (desc, node), slot in zip(marked_info, spec["slots"]):
                _append_depth(node, desc, slot["tag"], slot)
        else:
            for node in sampled:
                _append_depth(node, node.tag, node.tag)

        if len(tags_legacy) < 2:
            return None

        spec = {**spec, "slots": kept_slots}
        order = np.argsort(depths)
        sorted_legacy = [tags_legacy[i] for i in order]
        sorted_sem = [tags_sem[i] for i in order]
        return tags_legacy, tags_sem, sorted_sem, processed_image, spec

    def _prepare_marked_data(self, image, depth_map, nodes, qtype):
        """Top-level entry: choose annotation mode, produce depth-sorted tags.

        ~10% chance of random pixel sampling; otherwise object-based marking.
        MCQ format requires >= 4 tags; returns None if insufficient.

        Returns (legacy_tags, semantic_tags, sorted_sem, image_bytes, t_label, mark_spec)
        or None.
        """
        mcq_n = 4 if qtype == QuestionType.MCQ else None

        if rng().random() < 0.1:
            result = self._sample_random_points(
                depth_map, num_points=mcq_n or rng().randint(4, 7),
            )
            if result is not None:
                tags, sorted_tags, image_bytes, mark_spec = result
                if qtype == QuestionType.MCQ and len(tags) != 4:
                    return None
                return tags, tags, sorted_tags, image_bytes, "points:", mark_spec

        result = self._mark_and_sort(
            image, depth_map, nodes, num_objects=mcq_n,
        )
        if result is None:
            return None
        tags_legacy, tags_sem, sorted_sem, image_bytes, mark_spec = result
        if qtype == QuestionType.MCQ and len(tags_legacy) != 4:
            return None
        return tags_legacy, tags_sem, sorted_sem, image_bytes, "objects:", mark_spec

    def _prepare(self, graph):
        """Extract depth_map, optional RGB (render only), and mask-bearing nodes."""
        view = graph.primary_view
        depth_map = view.depth_map
        image = view.image if self.emit_marked_images else None
        if image is not None:
            assert image.size == depth_map.shape[::-1], (
                f"Image {image.size} vs depth_map {depth_map.shape[::-1]} dimension mismatch."
            )

        nodes = [n for n in graph.get_object_nodes()
                 if n.view_appearances.get(0) and n.view_appearances[0].mask_path]
        if not nodes:
            return None
        return depth_map, image, nodes

    # ── prompt generation ────────────────────────────────────────────

    def _build_ordering_prompt(self, depth_map, image, nodes, qtype, sub_task):
        """Build a depth-ordering QA pair."""
        marked = self._prepare_marked_data(image, depth_map, nodes, qtype)
        if marked is None:
            return None, None
        tags, tags_sem, sorted_sem, image_bytes, t_label, mark_spec = marked
        qtype_str = "OE" if qtype == QuestionType.OPEN_ENDED else "MCQ"
        is_points = t_label.startswith("points")

        if qtype == QuestionType.OPEN_ENDED:
            mode = pick_instruction_mode(_DEPTH_INSTRUCTION_MODES)
            tpl = f"depth.ordering.{mode}"
            prompt = self.render_structured_prompt(
                tpl,
                shared={
                    "T": t_label,
                    "A": ', '.join(tags),
                    "X": ', '.join(sorted_sem),
                },
            )
            self._record_turn(
                sub_task, tpl, prompt, qtype,
                mark_spec=mark_spec,
                coord_tags=tags_sem if is_points else None,
                sorted_semantic=sorted_sem,
            )
        else:
            mode = pick_instruction_mode(_DEPTH_INSTRUCTION_MODES)
            tpl = f"depth.ordering_mcq.{mode}"
            wrong_perms = []
            for _ in range(50):
                perm = list(sorted_sem)
                rng().shuffle(perm)
                if perm != sorted_sem and perm not in wrong_perms:
                    wrong_perms.append(perm)
                    if len(wrong_perms) == 3:
                        break
            if len(wrong_perms) < 3:
                return None, None

            candidates = [sorted_sem] + wrong_perms
            shuffled, answer_option = self._shuffle_mcq(candidates)
            options = [f"{'ABCD'[i]}:{str(list(shuffled[i]))}" for i in range(4)]

            prompt = self.render_structured_prompt(
                tpl,
                shared={"T": t_label, "Y": '\n'.join(options)},
                q_args={"X": ', '.join(tags)},
                a_args={"X": answer_option},
            )
            self._record_turn(
                sub_task, tpl, prompt, qtype,
                mark_spec=mark_spec,
                coord_tags=tags_sem if is_points else None,
            )
        return prompt, image_bytes

    def _build_choice_prompt(self, depth_map, image, nodes, qtype, sub_task):
        """Build a depth-choice QA pair."""
        marked = self._prepare_marked_data(image, depth_map, nodes, qtype)
        if marked is None:
            return None, None
        tags, tags_sem, sorted_sem, image_bytes, t_label, mark_spec = marked
        qtype_str = "OE" if qtype == QuestionType.OPEN_ENDED else "MCQ"
        is_points = t_label.startswith("points")

        r = rng().random()
        question_type = "farthest" if r < 0.4 else ("closest" if r < 0.8 else "choice")
        base = f"depth.{question_type}" + ("_mcq" if qtype == QuestionType.MCQ else "")
        mode = pick_instruction_mode(_DEPTH_INSTRUCTION_MODES)
        tpl_name = f"{base}.{mode}"
        obj_str = ', '.join(tags)

        if question_type == "farthest":
            correct_idx = len(sorted_sem) - 1
        elif question_type == "closest":
            correct_idx = 0
        else:
            correct_idx = rng().randint(0, len(sorted_sem) - 1)

        if qtype == QuestionType.OPEN_ENDED:
            shared = {"T": t_label, "A": obj_str, "X": str(sorted_sem[correct_idx])}
            if question_type == "choice":
                shared["B"] = ORDINALS[correct_idx]
            prompt = self.render_structured_prompt(tpl_name, shared=shared)
            self._record_turn(
                sub_task, tpl_name, prompt, qtype,
                mark_spec=mark_spec,
                type_label=t_label,
                sorted_semantic=[sorted_sem[correct_idx]],
                coord_tags=tags_sem if is_points else None,
            )
        else:
            wrong_idx = [i for i in range(len(sorted_sem)) if i != correct_idx]
            candidates = [sorted_sem[correct_idx]] + [sorted_sem[i] for i in rng().sample(wrong_idx, 3)]
            shuffled, answer_option = self._shuffle_mcq(candidates)
            options = [f"{'ABCD'[i]}:{str(shuffled[i])}" for i in range(4)]

            shared = {"T": t_label}
            if question_type == "choice":
                shared["Y"] = ORDINALS[correct_idx]
                shared["Z"] = '\n'.join(options)
            else:
                shared["Y"] = '\n'.join(options)

            prompt = self.render_structured_prompt(
                tpl_name, shared=shared,
                q_args={"X": obj_str}, a_args={"X": answer_option},
            )
            self._record_turn(
                sub_task, tpl_name, prompt, qtype,
                mark_spec=mark_spec,
                type_label=t_label,
                answer_extra={"answer": answer_option},
                coord_tags=tags_sem if is_points else None,
            )
        return prompt, image_bytes

    # ── handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _dispatch(self, graph, task_kind, qtype, sub_task):
        """Shared handler logic: prepare graph �?build prompt �?return result."""
        prepared = self._prepare(graph)
        if prepared is None:
            return None
        depth_map, image, nodes = prepared
        builder = self._build_ordering_prompt if task_kind == "ordering" else self._build_choice_prompt
        prompt, image_bytes = builder(depth_map, image, nodes, qtype, sub_task)
        if prompt is None:
            return None
        return prompt, image_bytes, qtype

    def _generate_depth_ordering_oe(self, graph):
        return self._dispatch(graph, "ordering", QuestionType.OPEN_ENDED, "depth_ordering_oe")

    def _generate_depth_ordering_mcq(self, graph):
        return self._dispatch(graph, "ordering", QuestionType.MCQ, "depth_ordering_mcq")

    def _generate_depth_choice_oe(self, graph):
        return self._dispatch(graph, "choice", QuestionType.OPEN_ENDED, "depth_choice_oe")

    def _generate_depth_choice_mcq(self, graph):
        return self._dispatch(graph, "choice", QuestionType.MCQ, "depth_choice_mcq")

