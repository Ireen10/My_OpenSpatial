"""
Multiview object position: cross-view relative direction (OE + MCQ).

Frames:
  object_relative  — premise: A relative to anchor B in image 1 (cardinal)
  viewer_at_anchor — premise: viewer at B, A on the [X] side (egocentric)

Answer modes: OE sentence|free; MCQ direct|sentence|free
"""

import math
import random
from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.prompt_template import PromptTemplate
from .core.visual_marker import MarkConfig
from utils.box_utils import check_box_3d_vertical_overlap
from .core.question_type import QuestionType
from ..prompt_templates.multiview_object_position_templates import FRAME_PREMISE_POOLS

from utils.image_utils import convert_pil_to_bytes


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Multiview Position"
    SUB_TASKS = {
        "position_oe":  {"default": 1, "handler": "_generate_position_oe"},
        "position_mcq": {"default": 1, "handler": "_generate_position_mcq"},
    }

    def get_mark_config(self):
        return MarkConfig(mark_types=["mask", "box"])

    _ANGLES = [0, 45, 90, 135, 180, -135, -90, -45]
    _DIR_FRAMES = ("object_relative", "viewer_at_anchor")
    _DIR_TEMPLATES = {
        "object_relative": [
            "north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest",
        ],
        "viewer_at_anchor": [
            "front", "front-right", "right", "back-right",
            "back", "back-left", "left", "front-left",
        ],
    }
    _DIR_MAPS = {
        "object_relative": dict(zip(
            _DIR_TEMPLATES["object_relative"], _ANGLES,
        )),
        "viewer_at_anchor": dict(zip(
            _DIR_TEMPLATES["viewer_at_anchor"], _ANGLES,
        )),
    }
    _OE_ANSWER_MODES = ("sentence", "free")
    _MCQ_ANSWER_MODES = ("direct", "sentence", "free")

    # ─── Direction Logic ──────────────────────────────────────────────

    @staticmethod
    def get_direction(dx, dy, dir_tmp, delta=15):
        angle = math.degrees(math.atan2(dx, dy))
        if -delta < angle <= delta:
            return dir_tmp[0]
        elif delta < angle <= 90-delta:
            return dir_tmp[1]
        elif 90-delta < angle <= 90+delta:
            return dir_tmp[2]
        elif 90+delta < angle <= 180-delta:
            return dir_tmp[3]
        elif angle > 180-delta or angle <= -180+delta:
            return dir_tmp[4]
        elif -180+delta < angle <= -90-delta:
            return dir_tmp[5]
        elif -90-delta < angle <= -90+delta:
            return dir_tmp[6]
        elif -90+delta < angle <= -delta:
            return dir_tmp[7]
        else:
            return 'unknown'

    @staticmethod
    def rotate(dx, dy, dx1, dy1, prior_direction, dir_map):
        actual_angle = math.degrees(math.atan2(dx1, dy1))
        target_angle = dir_map.get(prior_direction, 0)
        rotate_angle = target_angle - actual_angle
        rad = math.radians(-rotate_angle)
        new_dx = dx * math.cos(rad) - dy * math.sin(rad)
        new_dy = dx * math.sin(rad) + dy * math.cos(rad)
        return new_dx, new_dy

    def relative_direction(self, p1, p2, p3, frame):
        dir_tmp = self._DIR_TEMPLATES[frame]
        dir_map = self._DIR_MAPS[frame]
        prior_direction = dir_tmp[random.choice([0, 2, 4, 6])]
        dx1, dy1 = p1[0] - p2[0], p1[1] - p2[1]
        dx3, dy3 = p3[0] - p2[0], p3[1] - p2[1]
        new_dx3, new_dy3 = self.rotate(dx3, dy3, dx1, dy1, prior_direction, dir_map)
        level1_direction = self.get_direction(new_dx3, new_dy3, dir_tmp, delta=10)
        level2_direction = self.get_direction(new_dx3, new_dy3, dir_tmp, delta=15)
        level3_direction = self.get_direction(new_dx3, new_dy3, dir_tmp, delta=20)
        return prior_direction, [level1_direction, level2_direction, level3_direction]

    # ─── Prompt Functions ─────────────────────────────────────────────

    @classmethod
    def _pick_answer_mode(cls, question_type: QuestionType) -> str:
        modes = (
            cls._MCQ_ANSWER_MODES
            if question_type == QuestionType.MCQ
            else cls._OE_ANSWER_MODES
        )
        return random.choice(modes)

    def _vocab_for_direction(self, direction):
        if direction in self._DIR_TEMPLATES["object_relative"]:
            return self._DIR_TEMPLATES["object_relative"]
        return self._DIR_TEMPLATES["viewer_at_anchor"]

    def _build_mcq_options(self, dir_B2anchor, dir_B2anchors):
        """Build MCQ options string; answer token is always label + option text."""
        dir_tmp = self._vocab_for_direction(dir_B2anchor)
        wrong_options = [d for d in dir_tmp if d != dir_B2anchor and d not in dir_B2anchors]
        wrong_options = random.sample(wrong_options, 3)
        candidates = [dir_B2anchor] + wrong_options
        shuffled, answer_letter = self._shuffle_mcq(candidates)

        options_list = [f"{'ABCD'[i]}.{shuffled[i]}" for i in range(4)]
        options_str = "Options: " + " ".join(options_list) + "."
        answer_option = f"{answer_letter}.{dir_B2anchor}"
        return options_str, answer_option

    def position_prompt_func(self, A, B, anchor, question_type=QuestionType.OPEN_ENDED):
        """Generate a position QA (open-ended or MCQ).

        Args:
            A: (desc, box_3d_world) for object in View 1.
            B: (desc, box_3d_world) for object in View 2.
            anchor: (desc, box_3d_world) for the anchor object visible in both views.
            question_type: QuestionType.OPEN_ENDED or QuestionType.MCQ.

        Returns:
            Formatted "question Answer: answer" string, or None if direction is unknown.
        """
        A_desc, A_box_3d_world = A
        B_desc, B_box_3d_world = B
        anchor_desc, anchor_box_3d_world = anchor

        frame = random.choice(self._DIR_FRAMES)
        dir_A2anchor, dir_B2anchors = self.relative_direction(
            A_box_3d_world[:2], anchor_box_3d_world[:2], B_box_3d_world[:2], frame,
        )
        if dir_A2anchor == 'unknown' or 'unknown' in dir_B2anchors:
            return None

        dir_B2anchor = dir_B2anchors[random.choice([1, 2])]
        answer_mode = self._pick_answer_mode(question_type)
        tpl = (
            f"multiview_position.{frame}"
            if question_type == QuestionType.OPEN_ENDED
            else f"multiview_position.{frame}_mcq"
        )
        tpl_obj = self.get_structured_template(tpl)
        stem_i = random.randrange(len(tpl_obj.stem))
        premise = PromptTemplate._fill(FRAME_PREMISE_POOLS[frame][stem_i], {
            "A": A_desc, "B": anchor_desc, "C": B_desc, "X": dir_A2anchor,
        })
        shared = {
            "A": A_desc, "B": anchor_desc, "C": B_desc, "X": dir_A2anchor,
            "P": premise,
            "D": dir_B2anchor,
            "T": dir_B2anchor,
        }
        if question_type == QuestionType.MCQ:
            options_str, answer_option = self._build_mcq_options(dir_B2anchor, dir_B2anchors)
            shared["T"] = answer_option
            shared["E"] = answer_option.split(".", 1)[0]
            shared["O"] = options_str

        prompt = self.render_structured_prompt(
            tpl,
            instruction_type=answer_mode,
            stem_index=stem_i,
            shared=shared,
        )
        return prompt, tpl

    # ─── Data Finder ──────────────────────────────────────────────────

    def _find_pair_from_overlapping_views(self, graph):
        """Find anchor (in both views) + 2 unique objects (one per view).

        Uses _find_overlapping_views to locate an anchor node in two views,
        then finds objects unique to each view for position reasoning.

        Returns:
            (meta_data, True) or (None, False).
        """
        anchor_node, views = self._find_overlapping_views(graph, num_views=2, pose_diversity=False)
        if anchor_node is None:
            return None, False
        view1_idx, view2_idx = views

        # Find tags unique to each view (single traversal)
        view_data = self._tags_and_nodes_in_views(graph, [view1_idx, view2_idx])
        tags_v1 = set(view_data[view1_idx].keys())
        tags_v2 = set(view_data[view2_idx].keys())
        only_in_v1 = tags_v1 - tags_v2
        only_in_v2 = tags_v2 - tags_v1
        if not only_in_v1 or not only_in_v2:
            return None, False

        obj1_tag = random.choice(list(only_in_v1))
        obj2_tag = random.choice(list(only_in_v2))

        node1 = view_data[view1_idx][obj1_tag]
        node2 = view_data[view2_idx][obj2_tag]

        # Validate all 3D boxes exist and have no vertical overlap
        anchor_box_3d_world = anchor_node.box_3d_world
        if anchor_box_3d_world is None or node1.box_3d_world is None or node2.box_3d_world is None:
            return None, False
        if check_box_3d_vertical_overlap([anchor_box_3d_world, node1.box_3d_world, node2.box_3d_world]):
            return None, False

        app1 = node1.view_appearances[view1_idx]
        app2 = node2.view_appearances[view2_idx]
        anchor_app1 = anchor_node.view_appearances[view1_idx]

        meta = {
            "image": [graph.views[view1_idx].image, graph.views[view2_idx].image],
            "mask": [app1.mask, app2.mask],
            "tag": [obj1_tag, obj2_tag],
            "view_idx": [view1_idx, view2_idx],
            "bbox_2d": [app1.bbox_2d, app2.bbox_2d],
            "box_3d_world": [node1.box_3d_world, node2.box_3d_world],
            "node": [node1, node2],
            "anchor_tag": anchor_node.tag,
            "anchor_box_3d_world": anchor_box_3d_world,
            "anchor_bbox_2d": anchor_app1.bbox_2d,
            "anchor_mask": anchor_app1.mask,
            "anchor_node": anchor_node,
        }
        return meta, True

    # ─── Marking + Prompt ─────────────────────────────────────────────

    def _build_pair_position_prompt(self, graph, question_type):
        for _ in range(20):
            meta, flag = self._find_pair_from_overlapping_views(graph)
            if flag:
                break
        if not flag:
            return None

        mark_type = self.marker.choose_mark_type()
        A_prompt = (meta["tag"][0], meta["box_3d_world"][0])
        B_prompt = (meta["tag"][1], meta["box_3d_world"][1])
        anchor_prompt = (meta["anchor_tag"], meta["anchor_box_3d_world"])

        if random.random() < 0.5:
            A_obj = (meta["tag"][0], meta["node"][0], meta["bbox_2d"][0], meta["mask"][0])
            anchor_obj = (
                meta["anchor_tag"], meta["anchor_node"],
                meta["anchor_bbox_2d"], meta["anchor_mask"],
            )
            B_obj = (meta["tag"][1], meta["node"][1], meta["bbox_2d"][1], meta["mask"][1])

            self.marker.reset(shuffle=True)
            vi0, vi1 = meta["view_idx"][0], meta["view_idx"][1]
            processed_image1, info1 = self.plan_mark_for_qa(
                meta["image"][0],
                objs=[A_obj, anchor_obj],
                mark_type=mark_type,
                view_idx=vi0,
            )
            from .core.mark_spec import assemble_per_view_mark_spec
            s0 = self.marker.last_mark_spec
            processed_image2, info2 = self.plan_mark_for_qa(
                meta["image"][1],
                objs=[B_obj],
                mark_type=mark_type,
                view_idx=vi1,
            )
            s1 = self.marker.last_mark_spec
            row = getattr(self._thread_local, "preprocess_row", None)
            refs = self._qa_image_refs(row, meta)
            merged = assemble_per_view_mark_spec([
                {
                    "view_index": 0,
                    "image_ref": refs[0] if refs else None,
                    "mark_kinds": (s0 or {}).get("mark_kinds", []),
                    "slots": (s0 or {}).get("slots", []),
                },
                {
                    "view_index": 1,
                    "image_ref": refs[1] if len(refs) > 1 else None,
                    "mark_kinds": (s1 or {}).get("mark_kinds", []),
                    "slots": (s1 or {}).get("slots", []),
                },
            ])
            if merged is not None:
                self.marker._last_mark_spec = merged
            slot_infos = [info1[0], info1[1], info2[0]]
        else:
            processed_image1 = {"bytes": convert_pil_to_bytes(meta["image"][0])}
            processed_image2 = {"bytes": convert_pil_to_bytes(meta["image"][1])}
            slot_infos = [
                (meta["tag"][0], meta["node"][0]),
                (meta["anchor_tag"], meta["anchor_node"]),
                (meta["tag"][1], meta["node"][1]),
            ]

        result = self.position_prompt_func(A_prompt, B_prompt, anchor_prompt, question_type)
        if result is None:
            return None
        prompt, tpl = result
        sub = "position_mcq" if question_type == QuestionType.MCQ else "position_oe"
        self._record_turn(
            sub, tpl, prompt, question_type,
            mark_spec=self.marker.last_mark_spec,
            image_placeholder_count=2,
            view_indices=meta["view_idx"],
            extra_slots=self._slots_from_marked(slot_infos, labels=["A", "B", "C"]),
        )
        return prompt, [processed_image1, processed_image2]

    # ─── Handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _generate_position_oe(self, graph):
        result = self._build_pair_position_prompt(graph, question_type=QuestionType.OPEN_ENDED)
        if result is None:
            return None
        prompt, processed_images = result
        return prompt, processed_images, QuestionType.OPEN_ENDED

    def _generate_position_mcq(self, graph):
        result = self._build_pair_position_prompt(graph, question_type=QuestionType.MCQ)
        if result is None:
            return None
        prompt, processed_images = result
        return prompt, processed_images, QuestionType.MCQ
