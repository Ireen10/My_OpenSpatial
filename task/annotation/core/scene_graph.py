"""
Scene Graph data model for OpenSpatial annotation tasks.

Runtime-only representation — never serialized to parquet.
Built from a single parquet row (example dict) via factory class methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from functools import cached_property
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from PIL import Image

from utils.image_utils import load_depth_map
from utils.box_utils import convert_box_3d_world_to_camera


@dataclass
class ViewAppearance:
    """One object's appearance in a specific view."""
    mask_path: str = None
    bbox_2d: list = None                    # [x1, y1, x2, y2]
    pointcloud_camera_path: str = None      # path to .pcd file

    @cached_property
    def mask(self) -> Image.Image:
        if self.mask_path is None:
            return None
        return Image.open(self.mask_path)

    @cached_property
    def mask_array(self) -> np.ndarray:
        if self.mask is None:
            return None
        return np.array(self.mask)

    @cached_property
    def pointcloud_camera(self):
        if self.pointcloud_camera_path is None:
            return None
        import open3d as o3d
        return o3d.io.read_point_cloud(self.pointcloud_camera_path)


@dataclass
class ViewMeta:
    """Per-view camera and image metadata."""
    view_index: int
    image_path: str
    depth_map_path: str = None
    depth_scale: float = None
    pose_path: str = None
    intrinsic_path: str = None

    @cached_property
    def image(self) -> Image.Image:
        img = Image.open(self.image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    @cached_property
    def depth_map(self) -> np.ndarray:
        if self.depth_map_path is None:
            return None
        return load_depth_map(self.depth_map_path, self.depth_scale)

    @cached_property
    def pose(self) -> np.ndarray:
        if self.pose_path is None:
            return None
        return np.loadtxt(self.pose_path)

    @cached_property
    def intrinsic(self) -> np.ndarray:
        if self.intrinsic_path is None:
            return None
        return np.loadtxt(self.intrinsic_path)


@dataclass
class SceneNode:
    """An object in the scene, possibly visible across multiple views."""
    node_id: str                                      # singleview: f"{obj_idx}", multiview: str(box_3d)
    tag: str                                          # semantic label (e.g., "chair")
    view_appearances: Dict[int, ViewAppearance] = field(default_factory=dict)
    box_3d_world: list = None                         # [x,y,z, xl,yl,zl, roll,pitch,yaw]

    @cached_property
    def volume(self) -> float:
        if self.box_3d_world and len(self.box_3d_world) >= 6:
            return abs(self.box_3d_world[3] * self.box_3d_world[4] * self.box_3d_world[5])
        return 0.0

    @cached_property
    def center(self) -> np.ndarray:
        if self.box_3d_world and len(self.box_3d_world) >= 3:
            return np.array(self.box_3d_world[:3])
        return np.zeros(3)

    @cached_property
    def height(self) -> float:
        """Z-axis extent of the 3D bounding box."""
        if self.box_3d_world and len(self.box_3d_world) >= 6:
            return self.box_3d_world[5]
        return 0.0

    @cached_property
    def size(self) -> np.ndarray:
        """Size vector [xl, yl, zl]."""
        if self.box_3d_world and len(self.box_3d_world) >= 6:
            return np.array(self.box_3d_world[3:6])
        return np.zeros(3)

    @cached_property
    def rotation(self) -> list:
        """Rotation angles [roll, pitch, yaw]."""
        if self.box_3d_world and len(self.box_3d_world) >= 9:
            return self.box_3d_world[6:9]
        return [0.0, 0.0, 0.0]

    def box_3d_in_camera(self, pose: np.ndarray, euler_order: str = 'zxy') -> Optional[list]:
        """Convert world-frame 3D box to camera frame.

        Args:
            pose: 4x4 camera-to-world matrix.
            euler_order: Euler angle convention (default 'zxy').

        Returns:
            9-element list [x,y,z, xl,yl,zl, roll,pitch,yaw] in camera frame,
            or None if box_3d_world is unavailable.
        """
        return convert_box_3d_world_to_camera(self.box_3d_world, pose, euler_order)


@dataclass
class SceneGraph:
    """
    Runtime scene representation built from a parquet row.

    For singleview: one view, N nodes (one per detected object).
    For multiview: M views, nodes keyed by stringified 3D box (same physical
    object across views shares a node_id).
    """
    views: Dict[int, ViewMeta] = field(default_factory=dict)
    nodes: Dict[str, SceneNode] = field(default_factory=dict)

    # Multiview only: str(box_3d) → list of view indices
    box_to_view_proj: Dict[str, List[int]] = field(default_factory=dict)

    # Dataset metadata
    is_metric_depth: bool = False
    raw_example: dict = field(default_factory=dict, repr=False)

    @cached_property
    def duplicate_tags(self) -> Dict[str, int]:
        """Tags that appear more than once (for singleview: among obj_tags list)."""
        counts = Counter(node.tag for node in self.nodes.values())
        return {tag: count for tag, count in counts.items() if count > 1}

    @cached_property
    def node_list(self) -> List[SceneNode]:
        """Nodes in insertion order (matches original obj_tags order for singleview)."""
        return list(self.nodes.values())

    @cached_property
    def obj_tags(self) -> List[str]:
        """Tag list in node order (singleview: matches original obj_tags)."""
        return [node.tag for node in self.node_list]

    @property
    def primary_view(self) -> ViewMeta:
        """Return the first (or only) view. For singleview tasks this is view 0."""
        return next(iter(self.views.values()))

    def get_object_nodes(self, view_idx: int = None) -> List['SceneNode']:
        """Return nodes visible in the given view (defaults to primary view)."""
        if view_idx is None:
            view_idx = self.primary_view.view_index
        return [node for node in self.node_list
                if node.view_appearances.get(view_idx) is not None]

    def count_tag_in_view(self, tag: str, view_idx: int) -> int:
        """How many detected objects share ``tag`` in one view (for mark necessity)."""
        if not tag:
            return 0
        return sum(
            1 for node in self.get_object_nodes(view_idx)
            if node.tag == tag
        )

    def requires_mark_for_nodes(
        self,
        nodes: List['SceneNode'],
        view_idx: int = None,
    ) -> bool:
        """True when any referenced tag is ambiguous in the given view."""
        if not nodes:
            return False
        if view_idx is None:
            view_idx = self.primary_view.view_index
        for node in nodes:
            if self.count_tag_in_view(node.tag, view_idx) > 1:
                return True
        return False

    def get_overlapping_nodes(self, min_views: int = 2) -> List['SceneNode']:
        """Return nodes visible in at least min_views views (multiview)."""
        return [node for node in self.node_list if len(node.view_appearances) >= min_views]

    def get_node_view_pairs(self, node_id: str) -> List[int]:
        """Return view indices where this node is visible."""
        return self.box_to_view_proj.get(node_id, [])

    def sample_well_connected_box(self, min_views: int = 3, attempts: int = 5) -> Optional[Tuple[str, List[int]]]:
        """Find a random box visible in at least min_views views.

        Returns:
            (box_key, view_candidates) or None if not found.
        """
        import random
        box_keys = list(self.box_to_view_proj.keys())
        if not box_keys:
            return None
        for _ in range(attempts):
            box_key = random.choice(box_keys)
            view_candidates = self.box_to_view_proj[box_key]
            if len(view_candidates) >= min_views:
                return box_key, view_candidates
        return None

    # ─── Factory Methods ─────────────────────────────────────────────────

    @classmethod
    def from_singleview_example(cls, example: dict) -> 'SceneGraph':
        """
        Build a SceneGraph for singleview tasks from a parquet row.

        Expected columns: image, obj_tags, masks, bboxes_2d,
        and optionally: pointclouds, bboxes_3d_world_coords, depth_map,
        depth_scale, intrinsic, pose.
        """
        image_path = example.get("image")
        depth_map_path = example.get("depth_map")
        depth_scale = example.get("depth_scale")
        pose_path = example.get("pose")
        intrinsic_path = example.get("intrinsic")

        view = ViewMeta(
            view_index=0,
            image_path=image_path,
            depth_map_path=depth_map_path,
            depth_scale=depth_scale,
            pose_path=pose_path,
            intrinsic_path=intrinsic_path,
        )

        obj_tags = example.get("obj_tags", [])
        masks = example.get("masks", [])
        bboxes = example.get("bboxes_2d", [])
        pointclouds = example.get("pointclouds", [])
        boxes_3d = example.get("bboxes_3d_world_coords", [])

        nodes = {}
        for idx, tag in enumerate(obj_tags):
            node_id = str(idx)
            appearance = ViewAppearance(
                mask_path=masks[idx] if idx < len(masks) else None,
                bbox_2d=bboxes[idx] if idx < len(bboxes) else None,
                pointcloud_camera_path=pointclouds[idx] if idx < len(pointclouds) else None,
            )
            box_3d = boxes_3d[idx] if idx < len(boxes_3d) else None
            nodes[node_id] = SceneNode(
                node_id=node_id,
                tag=tag,
                view_appearances={0: appearance},
                box_3d_world=box_3d,
            )

        return cls(views={0: view}, nodes=nodes,
                   is_metric_depth=example.get("is_metric_depth", False),
                   raw_example=example)

    @classmethod
    def from_multiview_example(cls, example: dict, max_num_views: int = 400) -> 'SceneGraph':
        """
        Build a SceneGraph for multiview tasks from a parquet row.

        Expected columns: image (list), obj_tags (list of lists),
        masks (list of lists), bboxes_2d (list of lists),
        bboxes_3d_world_coords (list of lists),
        and optionally: depth_map, depth_scale, intrinsic, pose, pointclouds.
        """
        image_paths = example.get("image", [])
        num_views = len(image_paths)

        # Subsample views if exceeding max
        if num_views > max_num_views:
            samples_idx = np.linspace(0, num_views, num=max_num_views, endpoint=False, dtype=int).tolist()
        else:
            samples_idx = list(range(num_views))

        # Build ViewMeta for each sampled view
        views = {}
        for vi in samples_idx:
            views[vi] = ViewMeta(
                view_index=vi,
                image_path=image_paths[vi],
                depth_map_path=_safe_index(example.get("depth_map"), vi),
                depth_scale=_safe_index(example.get("depth_scale"), vi),
                pose_path=_safe_index(example.get("pose"), vi),
                intrinsic_path=_safe_index(example.get("intrinsic"), vi),
            )

        # Build nodes and box_to_view_proj
        nodes = {}
        box_to_view_proj = {}
        all_obj_tags = example.get("obj_tags", [])
        all_masks = example.get("masks", [])
        all_bboxes = example.get("bboxes_2d", [])
        all_pointclouds = example.get("pointclouds", [])
        all_boxes_3d = example.get("bboxes_3d_world_coords", [])

        for vi in samples_idx:
            view_tags = all_obj_tags[vi] if vi < len(all_obj_tags) else []
            view_masks = all_masks[vi] if vi < len(all_masks) else []
            view_bboxes = all_bboxes[vi] if vi < len(all_bboxes) else []
            view_pcds = all_pointclouds[vi] if all_pointclouds and vi < len(all_pointclouds) else []
            view_boxes_3d = all_boxes_3d[vi] if vi < len(all_boxes_3d) else []

            for obj_idx, tag in enumerate(view_tags):
                box_3d = view_boxes_3d[obj_idx] if obj_idx < len(view_boxes_3d) else None
                box_str = str(box_3d) if box_3d is not None else f"view{vi}_obj{obj_idx}"

                # Track box_to_view_proj
                if box_str not in box_to_view_proj:
                    box_to_view_proj[box_str] = []
                box_to_view_proj[box_str].append(vi)

                # Build appearance
                appearance = ViewAppearance(
                    mask_path=view_masks[obj_idx] if obj_idx < len(view_masks) else None,
                    bbox_2d=view_bboxes[obj_idx] if obj_idx < len(view_bboxes) else None,
                    pointcloud_camera_path=view_pcds[obj_idx] if view_pcds and obj_idx < len(view_pcds) else None,
                )

                # Create or update node
                if box_str not in nodes:
                    nodes[box_str] = SceneNode(
                        node_id=box_str,
                        tag=tag,
                        view_appearances={vi: appearance},
                        box_3d_world=box_3d,
                    )
                else:
                    nodes[box_str].view_appearances[vi] = appearance

        return cls(
            views=views,
            nodes=nodes,
            box_to_view_proj=box_to_view_proj,
            is_metric_depth=example.get("is_metric_depth", False),
            raw_example=example,
        )


def _safe_index(lst, idx):
    """Safely index into a list or return None."""
    if lst is None:
        return None
    if isinstance(lst, list) and idx < len(lst):
        return lst[idx]
    return None
