"""
Base class for multiview annotation tasks.

Extends BaseAnnotationTask with multiview-specific logic:
- multiview message creation (multiple <image> tags)
- backproject_2d_to_3d / project_3d_to_2d
"""

import random
from typing import List, Optional
import numpy as np
import open3d as o3d

from .base_annotation_task import BaseAnnotationTask
from .scene_graph import SceneGraph, SceneNode
from .mark_spec import assemble_per_view_mark_spec, merge_mark_specs
from .message_builder import create_multiview_messages

from utils.projection_utils import backproject_depth_to_3d, project_points_3d_to_2d, transform_points_camera_to_world
from utils.image_utils import convert_pil_to_bytes


class BaseMultiviewAnnotationTask(BaseAnnotationTask):
    """
    Base class for all multiview annotation tasks.

    Subclasses must implement:
        - process(self, example) -> (prompts, processed_images, question_tags, question_types)

    Provides shared multiview utilities:
        - backproject_2d_to_3d / project_3d_to_2d
    """

    QUESTION_TAG = "Unknown"

    def __init__(self, args):
        super().__init__(args)
        self.max_num_views = args.get("max_num_views", 400)
        self.min_rot_angle = args.get("min_rot_angle", 15.0)
        self.min_translation = args.get("min_translation", 0.0)
        if not getattr(self, "task_name", None):
            self.task_name = args.get("task_name") or args.get("file_name", "multiview")

    def build_scene_graph(self, example) -> SceneGraph:
        """Build a multiview SceneGraph."""
        return SceneGraph.from_multiview_example(example, max_num_views=self.max_num_views)

    def check_example(self, example) -> bool:
        """Validate required fields exist and have consistent lengths."""
        required_fields = [
            'image', 'pose', 'intrinsic', 'obj_tags', 'masks',
            'depth_map', 'bboxes_3d_world_coords',
        ]
        for field in required_fields:
            if field not in example:
                return False
        lengths = [len(example[field]) for field in required_fields]
        if len(set(lengths)) != 1:
            return False
        if lengths[0] == 0:
            return False
        return True

    def create_messages_from_prompts(self, prompts, processed_images):
        """Multiview variant: prepends N <image> tags based on image count."""
        return create_multiview_messages(prompts, processed_images)

    # ─── 2D ↔ 3D Mapping Utilities ───────────────────────────────────────

    def backproject_2d_to_3d(self, camera_to_world, depth, img_dim, intrinsic):
        """
        Back-project 2D image to 3D world coordinates using depth and camera params.

        Args:
            camera_to_world: 4x4 extrinsic matrix
            depth: H x W depth map
            img_dim: (width, height)
            intrinsic: 4x4 intrinsic matrix

        Returns:
            np.ndarray of shape (H*W, 3) — 3D world coordinates
        """
        return backproject_depth_to_3d(depth, img_dim, intrinsic, pose=camera_to_world)

    def project_3d_to_2d(self, world_to_camera, points_3d, intrinsic):
        """
        Project 3D world coordinates to 2D image coordinates.

        Args:
            world_to_camera: 4x4 extrinsic matrix (inverse of camera_to_world)
            points_3d: N x 3 array
            intrinsic: 4x4 intrinsic matrix

        Returns:
            np.ndarray of shape (N, 2) — (U, V) coordinates
        """
        return project_points_3d_to_2d(world_to_camera, points_3d, intrinsic)

    # ─── Shared Multiview Helpers ────────────────────────────────────────

    def _get_world_pointcloud(self, graph, node_id, view_idx):
        """Get world-frame pointcloud for a node in a given view via SceneGraph."""
        app = graph.nodes[node_id].view_appearances[view_idx]
        pcd_cam = app.pointcloud_camera
        pose = graph.views[view_idx].pose
        pcd_world_pts = transform_points_camera_to_world(np.asarray(pcd_cam.points), pose)
        pcd_world = o3d.geometry.PointCloud()
        pcd_world.points = o3d.utility.Vector3dVector(pcd_world_pts)
        return pcd_world

    def _tags_in_view(self, graph, view_idx):
        """Return set of tags visible in a given view."""
        return set(n.tag for n in graph.nodes.values() if view_idx in n.view_appearances)

    def _find_node_by_tag_in_view(self, graph, tag, view_idx):
        """Find first node with given tag visible in view."""
        for nid, node in graph.nodes.items():
            if node.tag == tag and view_idx in node.view_appearances:
                return nid, node
        return None, None

    def _tags_and_nodes_in_views(self, graph, view_indices):
        """Single-pass: build {view_idx: {tag: node}} for multiple views.

        More efficient than calling _tags_in_view + _find_node_by_tag_in_view
        separately when both tag sets and node lookups are needed.
        """
        result = {vi: {} for vi in view_indices}
        for node in graph.nodes.values():
            for vi in view_indices:
                if vi in node.view_appearances:
                    result[vi][node.tag] = node
        return result

    @staticmethod
    def _check_pose_diversity(pose, selected_poses, min_rot_angle, min_translation):
        """Check that a camera pose differs enough from all selected poses.

        Uses two complementary metrics:
        - Rotation: geodesic angle of relative rotation R1^T @ R2.
          Captures pan, tilt, and roll differences.
        - Translation: Euclidean distance between camera centers.

        A candidate is accepted only if it differs from EVERY selected pose
        on at least one metric (rotation OR translation).

        Args:
            pose: 4x4 camera-to-world matrix of the candidate view.
            selected_poses: list of 4x4 matrices already selected.
            min_rot_angle: minimum rotation angle in degrees.
            min_translation: minimum camera center distance.
                             Set 0 to disable translation check.
        """
        if pose is None:
            return True
        R_new = pose[:3, :3]
        t_new = pose[:3, 3]
        for existing_pose in selected_poses:
            R_ext = existing_pose[:3, :3]
            t_ext = existing_pose[:3, 3]
            # Geodesic rotation angle
            R_rel = R_new.T @ R_ext
            trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
            rot_angle = np.degrees(np.arccos(trace))
            # Translation distance
            trans_dist = np.linalg.norm(t_new - t_ext)
            # Reject if BOTH below thresholds (too similar on all axes)
            rot_ok = rot_angle >= min_rot_angle
            trans_ok = min_translation > 0 and trans_dist >= min_translation
            if not rot_ok and not trans_ok:
                return False
        return True

    def _raw_images_for_meta(self, meta) -> list:
        return [{"bytes": convert_pil_to_bytes(meta["image"][i])} for i in range(len(meta["image"]))]

    @staticmethod
    def _qa_image_refs(preprocess_row: Optional[dict], meta: dict) -> List[str]:
        """Pipeline image paths for each QA frame (ordered like meta['image'])."""
        if not preprocess_row or not isinstance(preprocess_row.get("image"), list):
            return []
        scene_paths = preprocess_row["image"]
        view_indices = meta.get("view_idx") or list(range(len(meta.get("image", []))))
        refs = []
        for i, vi in enumerate(view_indices):
            if vi < len(scene_paths):
                refs.append(str(scene_paths[vi]).replace("\\", "/"))
            else:
                refs.append("")
        return refs

    def _record_multiview_turn(
        self,
        sub_task: str,
        template_id: str,
        prompt: str,
        question_type,
        meta: dict,
        *,
        mark_spec: Optional[dict] = None,
        extra_slots: Optional[dict] = None,
    ):
        n_views = len(meta.get("image", []))
        view_indices = meta.get("view_idx")
        self._record_turn(
            sub_task, template_id, prompt, question_type,
            mark_spec=mark_spec,
            extra_slots=extra_slots,
            image_placeholder_count=n_views,
            view_indices=view_indices,
        )

    def _mark_per_view(self, meta, mark_type):
        """Mark one object per view image, return (processed_images, marked_infos)."""
        graph = getattr(self._thread_local, "scene_graph", None)
        processed_images = []
        marked_infos = []
        per_view_specs = []
        self.marker.reset(shuffle=True)
        row = getattr(self._thread_local, "preprocess_row", None)
        view_indices = meta.get("view_idx") or list(range(len(meta["image"])))
        image_refs = self._qa_image_refs(row, meta)
        nodes = meta.get("node")
        for i in range(len(meta["image"])):
            vi = view_indices[i] if i < len(view_indices) else i
            passthrough = nodes[i] if nodes and i < len(nodes) else meta["pointcloud"][i]
            obj = (meta["tag"][i], passthrough, meta["bbox_2d"][i], meta["mask"][i])
            mark_nodes = [passthrough] if isinstance(passthrough, SceneNode) else []
            enable = self.resolve_mark_enabled(
                graph,
                mark_nodes if mark_nodes else None,
                view_idx=vi,
                tag=meta["tag"][i] if not mark_nodes else None,
            )
            if not enable:
                per_view_specs.append({
                    "version": 2, "mark_kinds": [], "slots": [],
                })
                img = {"bytes": convert_pil_to_bytes(meta["image"][i])}
                processed_images.append(img)
                marked_infos.append((meta["tag"][i], passthrough))
                continue
            if self.emit_marked_images:
                img, info = self.marker.mark_objects(
                    meta["image"][i], [obj], mark_type=mark_type, view_idx=vi,
                    preprocess_row=row,
                )
                per_view_specs.append(self.marker.last_mark_spec)
            else:
                spec, info = self.marker.plan_mark(
                    [obj], mark_type=mark_type, view_idx=vi,
                )
                per_view_specs.append(spec)
                img = {"bytes": convert_pil_to_bytes(meta["image"][i])}
            processed_images.append(img)
            marked_infos.append(info[0])
        merged = merge_mark_specs(
            per_view_specs,
            view_indices=list(range(len(meta["image"]))),
            image_refs=image_refs,
        )
        if merged:
            self.marker._last_mark_spec = merged
        return processed_images, marked_infos

    def _resolve_qa_images(self, graph, processed_images: list) -> list:
        """Map handler outputs to QA_images (M3: unmarked bytes unless emit_marked_images)."""
        if self.emit_marked_images:
            return processed_images
        if not processed_images:
            return processed_images

        example = getattr(self._thread_local, "preprocess_row", None)
        scene_images = example.get("image") if example and isinstance(example.get("image"), list) else []
        scene_n = len(scene_images)

        def _to_bytes(im) -> dict:
            if isinstance(im, dict) and im.get("bytes") is not None:
                return im
            return {"bytes": convert_pil_to_bytes(im)}

        resolved = []
        for item in processed_images:
            if isinstance(item, list):
                # Per-QA view list (e.g. correspondence: 2 views). Do not expand to full scene.
                if scene_n and 0 < len(item) < scene_n:
                    resolved.append([_to_bytes(im) for im in item])
                    continue
                # Legacy: inner list spans entire scene (len == scene_n).
                if scene_n and len(item) == scene_n:
                    resolved.append([_to_bytes(im) for im in item])
                    continue
                resolved.append([_to_bytes(im) for im in item])
                continue
            if isinstance(item, dict):
                resolved.append(item)
            else:
                resolved.append(_to_bytes(item))

        if resolved:
            return resolved

        if scene_n and len(processed_images) == scene_n:
            return [_to_bytes(im) for im in scene_images]
        return super()._resolve_qa_images(graph, processed_images)

    def _find_chain_and_mark(self, graph, num_views, retries=5):
        """Retry _find_view_chain, build meta, mark per view.

        Common entry point for multiview size/distance handlers that follow
        the pattern: find chain → build meta → mark.

        Returns (meta, processed_images, marked_infos) or None.
        """
        node_views = None
        for _ in range(retries):
            node_views = self._find_view_chain(graph, num_views=num_views)
            if node_views is not None:
                break
        if node_views is None:
            return None
        meta = self._build_view_meta(graph, node_views)
        mark_type = self.marker.choose_mark_type()
        processed_images, marked_infos = self._mark_per_view(meta, mark_type)
        return meta, processed_images, marked_infos

    def _find_overlapping_views(self, graph, num_views=2, *, pose_diversity=True):
        """Find a node visible in multiple views with pose diversity.

        This is the meta-function for all multiview finders: it locates a
        well-connected node and selects diverse camera views where it appears.

        Args:
            graph: SceneGraph with multiview data.
            num_views: Number of views to select (default 2).
            pose_diversity: If True, enforce min_rot_angle / min_translation
                            between selected views. If False, random sample.

        Returns:
            (node, [view_indices]) or (None, None).
        """
        result = graph.sample_well_connected_box()
        if result is None:
            return None, None
        nid, view_candidates = result
        node = graph.nodes[nid]

        visible = [vi for vi in view_candidates if vi in node.view_appearances]

        if pose_diversity:
            selected_poses = []
            chosen = []
            shuffled = list(visible)
            random.shuffle(shuffled)
            for vi in shuffled:
                pose = graph.views[vi].pose
                if self._check_pose_diversity(pose, selected_poses,
                                              self.min_rot_angle, self.min_translation):
                    chosen.append(vi)
                    if pose is not None:
                        selected_poses.append(pose)
                    if len(chosen) == num_views:
                        break
        else:
            if len(visible) < num_views:
                return None, None
            chosen = random.sample(visible, num_views)

        if len(chosen) < num_views:
            return None, None
        return node, chosen

    def _find_view_chain(self, graph, num_views=2, *, pose_diversity=True):
        """Chain-walk through the scene graph to find N different objects in N views.

        Starting from a random box, walks through the graph: find a view for
        the current box, then pick a new box visible in that view as the next
        anchor, and repeat.

        Args:
            graph: SceneGraph with multiview data.
            num_views: Number of (node, view) pairs to collect (default 2).
            pose_diversity: If True, enforce min_rot_angle / min_translation
                            between selected views. If False, skip pose check.

        Returns:
            [(node, view_idx), ...] or None.
        """
        box_to_views = graph.box_to_view_proj
        if not box_to_views:
            return None

        # Pre-build view → node_ids index for O(1) lookup
        view_to_nodes = {}
        for nid, node in graph.nodes.items():
            for vi in node.view_appearances:
                view_to_nodes.setdefault(vi, []).append(nid)

        used_views = set()
        used_boxes = set()
        collected = []           # [(node, view_idx)]
        selected_poses = []      # cached pose matrices for diversity check

        box_keys = list(box_to_views.keys())
        anchor_box = random.choice(box_keys)
        used_boxes.add(anchor_box)

        while len(collected) < num_views:
            # Filter unused views for the current anchor
            candidates = [v for v in box_to_views.get(anchor_box, [])
                          if v not in used_views]
            # Enforce pose diversity against already-selected views
            if pose_diversity and selected_poses:
                min_rot = self.min_rot_angle
                min_trans = self.min_translation
                if min_rot > 0 or min_trans > 0:
                    candidates = [v for v in candidates
                                  if self._check_pose_diversity(
                                      graph.views[v].pose, selected_poses,
                                      min_rot, min_trans)]
            if not candidates:
                return None

            view_idx = random.choice(candidates)
            used_views.add(view_idx)

            # Cache pose for future diversity checks
            pose = graph.views[view_idx].pose
            if pose is not None:
                selected_poses.append(pose)

            node = graph.nodes[anchor_box]
            collected.append((node, view_idx))

            if len(collected) == num_views:
                break

            # Pick next anchor: a box in this view that still has unused views
            node_ids_in_view = view_to_nodes.get(view_idx, [])
            viable = [nid for nid in node_ids_in_view
                       if nid not in used_boxes
                       and any(v not in used_views
                               for v in box_to_views.get(nid, []))]
            if not viable:
                return None
            anchor_box = random.choice(viable)
            used_boxes.add(anchor_box)

        return collected

    _ALL_META_FIELDS = [
        "image", "mask", "tag", "view_idx", "pointcloud", "bbox_2d", "box_3d_world", "node",
    ]

    def _build_view_meta(self, graph, node_views, *, fields=None):
        """Build meta_dict from (node, view_idx) entries.

        Args:
            graph: SceneGraph with multiview data.
            node_views: [(node, view_idx), ...] from _find_view_chain, etc.
            fields: List of field names to include. Default: all fields.
                    Supported: image, mask, tag, view_idx, pointcloud,
                    bbox_2d, box_3d_world.

        Returns:
            meta_data dict with the requested fields as keys.
        """
        fields = fields if fields is not None else self._ALL_META_FIELDS
        meta = {f: [] for f in fields}

        for node, view_idx in node_views:
            app = node.view_appearances[view_idx]
            if "image" in meta:
                meta["image"].append(graph.views[view_idx].image)
            if "mask" in meta:
                meta["mask"].append(app.mask)
            if "tag" in meta:
                meta["tag"].append(node.tag)
            if "view_idx" in meta:
                meta["view_idx"].append(view_idx)
            if "pointcloud" in meta:
                meta["pointcloud"].append(self._get_world_pointcloud(graph, node.node_id, view_idx))
            if "bbox_2d" in meta:
                meta["bbox_2d"].append(app.bbox_2d)
            if "box_3d_world" in meta:
                meta["box_3d_world"].append(node.box_3d_world)
            if "node" in meta:
                meta["node"].append(node)

        return meta

    @staticmethod
    def _marked_prompt_items(meta: dict, marked_infos: list) -> list:
        """Map marked tuples to (semantic tag, metric passthrough) for prompt funcs."""
        from .sample_metadata import marked_surface_label

        items = []
        for i, item in enumerate(marked_infos):
            desc = marked_surface_label(item)
            passthrough = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
            if isinstance(passthrough, SceneNode):
                passthrough = meta["pointcloud"][i]
            items.append((desc, passthrough))
        return items
