"""
Base class for singleview annotation tasks.

Extracts the shared run/apply_transform/create_messages patterns
from all 8 singleview annotation files.
"""

import threading
from typing import Dict, List, Optional, Tuple, Any

from .scene_graph import SceneGraph, SceneNode
from .visual_marker import VisualMarker, MarkConfig
from .message_builder import create_singleview_messages
from .prompt_template import PromptRenderRecord, PromptTemplate, TemplateRegistry
from .structured_prompt_template import StructuredPromptTemplate, StructuredTemplateRegistry
from .sample_metadata import build_turn_record, build_visual_anchor, split_prompt_qa
from .mark_spec import mark_spec_has_slots, wrap_single_view_mark_spec
from .question_type import QuestionType

from task.base_task import BaseTask
from utils.data_utils import as_python_list, normalize_image_field
from utils.point_cloud_utils import clean_point_cloud
from utils.image_utils import convert_pil_to_bytes

import task.prompt_templates  # noqa: F401

# Distinguish "caller did not pass mark_spec" from "this turn has no marks".
_MARK_SPEC_UNSET = object()

# Unique in-view tags: enable mark with this probability (else mark_spec omitted).
OPTIONAL_MARK_ENABLE_PROB = 0.25


class BaseAnnotationTask(BaseTask):
    """
    Base class for all singleview annotation tasks.

    Subclasses must implement:
        - process(self, example) -> (prompts, processed_images, question_tags, question_types)

    Optionally override:
        - get_mark_config() -> MarkConfig
        - check_example(example) -> bool
        - create_messages_from_prompts(prompts, processed_images) -> list
    """

    QUESTION_TAG = "Unknown"
    SUB_TASKS = {}

    def __init__(self, args):
        super().__init__(args)
        self._thread_local = threading.local()
        self.scaling_factor = args.get("scaling_factor", 1)
        self.filter_tags = args.get("filter_tags", None)
        self._sub_tasks_config = self._parse_sub_tasks(args.get("sub_tasks", None))
        self.emit_metadata = bool(args.get("emit_metadata", True))
        self.emit_marked_images = bool(args.get("emit_marked_images", False))
        self.task_name = args.get("task_name") or args.get("file_name", "annotation")

    def _parse_sub_tasks(self, raw):
        if raw is None or raw == "all":
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list):
            return {k: None for k in raw}
        if hasattr(raw, '__dict__'):
            return vars(raw)
        raise ValueError(f"Invalid sub_tasks config: {raw}")

    def get_sub_task_count(self, sub_task, default=1):
        if self._sub_tasks_config is None:
            return default
        if sub_task not in self._sub_tasks_config:
            return 0
        count = self._sub_tasks_config[sub_task]
        return default if count is None else int(count)

    @property
    def marker(self):
        tl = self._thread_local
        if not hasattr(tl, 'marker'):
            tl.marker = VisualMarker(self.get_mark_config())
        return tl.marker

    @marker.setter
    def marker(self, value):
        self._thread_local.marker = value

    def get_mark_config(self) -> MarkConfig:
        return MarkConfig()

    @staticmethod
    def _qtype_str(qtype) -> str:
        if qtype == QuestionType.MCQ or str(qtype).upper() == "MCQ":
            return "MCQ"
        if qtype == QuestionType.JUDGMENT or str(qtype).lower() in ("judgment", "true_false"):
            return "Judgment"
        return "OE"

    def _init_example_context(self, example):
        self._thread_local.turn_records = []
        self._thread_local.viz_turns = []
        self._thread_local.preprocess_row = normalize_image_field(example)
        self._thread_local.last_prompt_render = None

    def _record_turn(
        self,
        sub_task: str,
        template_id: str,
        prompt: str,
        question_type,
        *,
        mark_spec: Any = _MARK_SPEC_UNSET,
        extra_slots: Optional[Dict[str, dict]] = None,
        type_label: str = "",
        sorted_semantic: Optional[List[str]] = None,
        answer_extra: Optional[Dict[str, str]] = None,
        coord_tags: Optional[List[str]] = None,
        instruction_mode: str = "legacy",
        question_prefix: Optional[str] = None,
        image_placeholder_count: int = 1,
        answer_text: Optional[str] = None,
        view_indices: Optional[list] = None,
    ):
        if not self.emit_metadata:
            return
        records = getattr(self._thread_local, "turn_records", None)
        if records is None:
            return
        if mark_spec is _MARK_SPEC_UNSET:
            mark_spec = self.marker.last_mark_spec
        if mark_spec and mark_spec.get("slots") and not mark_spec.get("views"):
            row = getattr(self._thread_local, "preprocess_row", None)
            ref = None
            img = as_python_list(row.get("image")) if row else None
            if isinstance(img, list):
                if len(img) == 1:
                    ref = str(img[0])
            elif img is not None:
                ref = str(img)
            mark_spec = wrap_single_view_mark_spec(mark_spec, image_ref=ref)
        render = getattr(self._thread_local, "last_prompt_render", None)
        if answer_text is None and " Answer: " in prompt:
            _, answer_text = split_prompt_qa(prompt)
        meta_turn, viz = build_turn_record(
            turn_id=len(records),
            task_name=self.task_name,
            sub_task=sub_task,
            question_type=self._qtype_str(question_type),
            template_id=template_id,
            prompt=prompt,
            render=render,
            mark_spec=mark_spec,
            referent_slots=extra_slots,
            extra_slots=extra_slots,
            sorted_semantic=sorted_semantic,
            coord_tags=coord_tags,
            instruction_mode=instruction_mode,
            question_prefix=question_prefix,
            image_placeholder_count=image_placeholder_count,
            answer_text=answer_text,
            type_label=type_label or None,
        )
        if view_indices is not None:
            meta_turn["view_indices"] = list(view_indices)
        records.append(meta_turn)
        viz_list = getattr(self._thread_local, "viz_turns", None)
        if viz_list is not None:
            viz_list.append(viz)

    @staticmethod
    def _slots_from_nodes(nodes: List[SceneNode], labels: Optional[List[str]] = None) -> Dict[str, dict]:
        slots = {}
        for i, node in enumerate(nodes):
            sid = labels[i] if labels and i < len(labels) else chr(ord("A") + i)
            try:
                oid = int(node.node_id)
            except (TypeError, ValueError):
                oid = i
            slots[sid] = {"obj_idx": oid, "tag": node.tag}
        return slots

    @staticmethod
    def _slots_from_marked(marked: list, labels: Optional[List[str]] = None) -> Dict[str, dict]:
        slots = {}
        for i, item in enumerate(marked):
            sid = labels[i] if labels and i < len(labels) else chr(ord("A") + i)
            if isinstance(item[1], SceneNode):
                node = item[1]
                tag = node.tag
                try:
                    oid = int(node.node_id)
                except (TypeError, ValueError):
                    oid = i
            else:
                tag = item[0].split("-(")[0] if "-(" in str(item[0]) else str(item[0])
                oid = i
            slots[sid] = {"obj_idx": oid, "tag": tag}
        return slots

    def render_and_record(
        self,
        template_name: str,
        sub_task: str,
        question_type,
        *,
        condition: bool = None,
        shared: dict = None,
        q_args: dict = None,
        a_args: dict = None,
        mark_spec: Optional[dict] = None,
        extra_slots: Optional[Dict[str, dict]] = None,
        **turn_kw,
    ) -> str:
        prompt = self.render_prompt(
            template_name, condition=condition,
            shared=shared, q_args=q_args, a_args=a_args,
        )
        self._record_turn(
            sub_task, template_name, prompt, question_type,
            mark_spec=mark_spec, extra_slots=extra_slots, **turn_kw,
        )
        return prompt

    @staticmethod
    def _get_cloud(marked):
        desc, node = marked
        cloud = node.view_appearances[0].pointcloud_camera
        return desc, cloud

    @staticmethod
    def _clean_cloud(cloud):
        return clean_point_cloud(cloud)

    @staticmethod
    def _shuffle_mcq(candidates, correct_idx=0):
        import random
        order = list(range(len(candidates)))
        random.shuffle(order)
        answer = "ABCD"[order.index(correct_idx)]
        return [candidates[j] for j in order], answer

    @staticmethod
    def _nodes_from_mark_inputs(objs) -> List[SceneNode]:
        nodes: List[SceneNode] = []
        if not objs:
            return nodes
        for item in objs:
            if isinstance(item, SceneNode):
                nodes.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                node = item[1]
                if isinstance(node, SceneNode):
                    nodes.append(node)
        return nodes

    @staticmethod
    def _marked_info_without_plan(objs) -> list:
        out = []
        for item in objs or []:
            if isinstance(item, SceneNode):
                out.append((item.tag, item))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((item[0], item[1]))
        return out

    def resolve_mark_enabled(
        self,
        graph: Optional[SceneGraph],
        objs=None,
        view_idx: int = 0,
        *,
        tag: Optional[str] = None,
    ) -> bool:
        """Require mark when tag is ambiguous in-view; else sample optional mark."""
        import random

        if graph is not None:
            nodes = self._nodes_from_mark_inputs(objs) if objs else []
            if nodes and graph.requires_mark_for_nodes(nodes, view_idx):
                return True
            check_tag = tag
            if not check_tag and nodes:
                check_tag = nodes[0].tag
            if check_tag and graph.count_tag_in_view(check_tag, view_idx) > 1:
                return True
        return random.random() < OPTIONAL_MARK_ENABLE_PROB

    def _qa_slot(self, image=None):
        """In-memory QA image slot; serialize pixels only when emit_marked_images."""
        if self.emit_marked_images and image is not None:
            return {"bytes": convert_pil_to_bytes(image)}
        return None

    def _qa_slots(self, images):
        if self.emit_marked_images:
            return [{"bytes": convert_pil_to_bytes(im)} for im in images]
        return [None] * len(images)

    @staticmethod
    def _qa_pixel_placeholders(items):
        """Preserve per-QA image counts without encoding pixels."""
        if not items:
            return items
        out = []
        for item in items:
            if isinstance(item, list):
                out.append([None] * len(item))
            else:
                out.append(None)
        return out

    def plan_mark_for_qa(
        self,
        image,
        *,
        objs=None,
        mark_type=None,
        labels=None,
        points=None,
        view_idx: int = 0,
    ):
        """Plan mark_spec; pixels only when emit_marked_images."""
        graph = getattr(self._thread_local, "scene_graph", None)
        if points is None and objs is not None:
            if not self.resolve_mark_enabled(graph, objs, view_idx):
                self.marker._last_mark_spec = None
                marked = self._marked_info_without_plan(objs)
                return self._qa_slot(image), marked

        row = getattr(self._thread_local, "preprocess_row", None)
        if self.emit_marked_images:
            if image is None:
                raise ValueError("image is required when emit_marked_images is true")
            from .mark_spec import render_mark

            if points is not None:
                spec, marked_info = self.marker.plan_mark(
                    points=points, labels=labels,
                )
            else:
                spec, marked_info = self.marker.plan_mark(
                    objs, mark_type=mark_type, view_idx=view_idx, labels=labels,
                )
            self.marker._last_mark_spec = spec
            rendered = render_mark(
                image, spec, labels=labels, preprocess_row=row, view_index=view_idx,
            )
            return rendered, marked_info

        if points is not None:
            spec, marked_info = self.marker.plan_mark(points=points, labels=labels)
        else:
            spec, marked_info = self.marker.plan_mark(
                objs, mark_type=mark_type, view_idx=view_idx, labels=labels,
            )
        self.marker._last_mark_spec = spec
        return self._qa_slot(image), marked_info

    def mark_objects_for_qa(
        self,
        image,
        objs,
        *,
        mark_type=None,
        labels=None,
        view_idx: int = 0,
    ):
        """``image`` is only read when ``emit_marked_images`` is true (pixel render)."""
        return self.plan_mark_for_qa(
            image,
            objs=objs,
            mark_type=mark_type,
            labels=labels,
            view_idx=view_idx,
        )

    def _resolve_qa_images(self, graph, processed_images: list) -> list:
        if self.emit_marked_images:
            return processed_images
        return self._qa_pixel_placeholders(processed_images)

    def check_example(self, example) -> bool:
        if "image" not in example:
            return False
        if "obj_tags" not in example or len(example["obj_tags"]) == 0:
            return False
        return True

    def build_scene_graph(self, example) -> SceneGraph:
        return SceneGraph.from_singleview_example(example)

    def process(self, graph, example):
        prompts, images, qtypes = [], [], []
        for name, meta in self.SUB_TASKS.items():
            count = self.get_sub_task_count(name, default=meta["default"])
            if count == 0:
                continue
            handler = getattr(self, meta["handler"])
            for _ in range(count):
                result = handler(graph)
                if result is None:
                    continue
                if isinstance(result, list):
                    for p, img, qt in result:
                        prompts.append(p)
                        images.append(img)
                        qtypes.append(qt)
                else:
                    p, img, qt = result
                    prompts.append(p)
                    images.append(img)
                    qtypes.append(qt)
        tags = [[self.QUESTION_TAG]] * len(prompts)
        return prompts, images, tags, qtypes

    def get_template(self, name: str) -> PromptTemplate:
        return TemplateRegistry.get(name)

    def render_prompt(self, template_name: str, condition: bool = None, *,
                      shared: dict = None, q_args: dict = None, a_args: dict = None) -> str:
        tpl = self.get_template(template_name)
        rec = tpl.render_provenance(
            template_name,
            condition=condition,
            shared=shared,
            q_args=q_args,
            a_args=a_args,
        )
        self._thread_local.last_prompt_render = rec
        return rec.to_prompt()

    def render_prompt_with_options(
        self,
        template_name: str,
        options_suffix: str,
        *,
        condition: bool = None,
        shared: dict = None,
        q_args: dict = None,
        a_args: dict = None,
    ) -> str:
        """MCQ helper: sample template, append options before Answer, keep provenance in sync."""
        tpl = self.get_template(template_name)
        rec = tpl.render_provenance(
            template_name,
            condition=condition,
            shared=shared,
            q_args=q_args,
            a_args=a_args,
        )
        self._thread_local.last_prompt_render = rec
        return rec.question_text + options_suffix + " Answer: " + rec.answer_text

    def get_structured_template(self, template_id: str) -> StructuredPromptTemplate:
        return StructuredTemplateRegistry.get(template_id)

    def render_structured_prompt(
        self,
        template_id: str,
        *,
        condition: bool = None,
        instruction_type: str = None,
        stem_index: int = None,
        shared: dict = None,
        q_args: dict = None,
        a_args: dict = None,
        is_metric_depth: bool = None,
    ) -> str:
        """M8: render from introduction/stem/q_instruction + answer instruction_type pools."""
        tpl = self.get_structured_template(template_id)
        rec = tpl.render_provenance(
            condition=condition,
            instruction_type=instruction_type,
            stem_index=stem_index,
            shared=shared,
            q_args=q_args,
            a_args=a_args,
            is_metric_depth=is_metric_depth,
        )
        self._thread_local.last_prompt_render = rec
        return rec.to_prompt()

    def render_structured_with_options(
        self,
        template_id: str,
        options_suffix: str,
        *,
        condition: bool = None,
        instruction_type: str = None,
        stem_index: int = None,
        shared: dict = None,
        q_args: dict = None,
        a_args: dict = None,
        is_metric_depth: bool = None,
    ) -> str:
        """MCQ: sample structured template, append options before Answer."""
        tpl = self.get_structured_template(template_id)
        rec = tpl.render_provenance(
            condition=condition,
            instruction_type=instruction_type,
            stem_index=stem_index,
            shared=shared,
            q_args=q_args,
            a_args=a_args,
            is_metric_depth=is_metric_depth,
        )
        self._thread_local.last_prompt_render = rec
        return rec.question_text + options_suffix + " Answer: " + rec.answer_text

    def create_messages_from_prompts(self, prompts, processed_images=None):
        return create_singleview_messages(prompts)

    def apply_transform(self, example, idx=None):
        if not self.check_example(example):
            return None, False

        self._init_example_context(example)
        self.marker = VisualMarker(self.get_mark_config())

        graph = self.build_scene_graph(example)
        self._thread_local.scene_graph = graph
        prompts, processed_images, question_tags, question_types = self.process(graph, example)
        if len(prompts) == 0:
            return None, False

        processed_images = self._resolve_qa_images(graph, processed_images)
        viz_turns = getattr(self._thread_local, "viz_turns", []) or []
        turn_records = getattr(self._thread_local, "turn_records", []) or []

        from .message_placeholders import (
            build_messages_from_viz_turns,
            sync_messages_with_qa_images,
            sync_messages_with_turns,
        )

        if viz_turns:
            messages = build_messages_from_viz_turns(viz_turns)
        else:
            messages = self.create_messages_from_prompts(prompts, processed_images)
            if turn_records:
                messages = sync_messages_with_turns(messages, turn_records)
            else:
                messages = sync_messages_with_qa_images(messages, processed_images)

        example["messages"] = messages
        example["question_tags"] = question_tags
        example["question_types"] = question_types

        if self.emit_metadata:
            turn_records = getattr(self._thread_local, "turn_records", [])
            for i, tr in enumerate(turn_records):
                tr["turn_id"] = i
            example["metadata"] = []
            from .sample_metadata import build_turn_metadata

            for tr in turn_records:
                example["metadata"].append(build_turn_metadata(example, tr))

        return example, True
