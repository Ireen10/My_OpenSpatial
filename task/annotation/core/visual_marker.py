"""
Unified visual marking utility for OpenSpatial annotation tasks.

Replaces the duplicated mark_objects() across 10+ files.
Each task configures its own MarkConfig with different mark types and weights.

Drawing primitives (draw_masks_on_image, draw_boxes_on_image, draw_points_on_image)
are co-located here as they are only used by VisualMarker.
"""

import os
from .thread_rng import rng
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from utils.image_utils import convert_pil_to_bytes


# ─── Constants ────────────────────────────────────────────────────────────────

COLOR_QUEUE_DEFAULT = ["red", "blue", "green", "pink", "yellow", "orange", "purple", "brown"]

COLOR_MAP = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "pink": (255, 192, 203),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "brown": (165, 42, 42),
}

_FONT_CANDIDATES = [
    os.environ.get("OPENSPATIAL_FONT", ""),
    "/pfs/jianhuiliu/fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
]


def _load_font(size):
    """Load a TrueType font, falling back through candidates then to default."""
    for path in _FONT_CANDIDATES:
        if path and os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ─── Label Drawing Helper ────────────────────────────────────────────────────

def _draw_labels(draw, anchors, labels, colors, img_size):
    """Draw text labels with colored background rectangles at anchor positions.

    Each label is placed near its anchor point, with automatic repositioning
    when the label would overflow the image boundary. Background color matches
    the marker; text color auto-selects black/white based on luminance.

    Args:
        draw:     PIL.ImageDraw object to draw on.
        anchors:  list of (x, y) anchor positions (one per label).
        labels:   list of str labels.
        colors:   list of (R, G, B) tuples.
        img_size: (width, height) of the image.
    """
    img_w, img_h = img_size
    scale_factor = min(img_w / 640, img_h / 480)
    font_size = max(12, int(24 * scale_factor))
    padding = max(2, int(3 * scale_factor))
    radius = max(2, int(6 * scale_factor))
    font = _load_font(font_size)

    for (ax, ay), label, color in zip(anchors, labels, colors):
        try:
            bbox = font.getbbox(label)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(label)

        # Default: top-right of anchor; shift if near edges
        lx = ax + radius + 4
        ly = ay - radius - text_h - padding
        if lx + text_w + padding * 2 > img_w:
            lx = ax - radius - 4 - text_w - padding * 2
        if lx < 0:
            lx = 0
        if ly < 0:
            ly = ay + radius + 4
        if ly + text_h + padding * 2 > img_h:
            ly = img_h - text_h - padding * 2

        bg_rect = [lx, ly, lx + text_w + padding * 2, ly + text_h + padding * 2]
        draw.rectangle(bg_rect, fill=color)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
        draw.text((lx + padding, ly + padding), label, fill=text_color, font=font)


# ─── Drawing Primitives ──────────────────────────────────────────────────────

def draw_masks_on_image(image, masks, colors, labels=None):
    """Draw semi-transparent colored masks with contour outlines on an image.

    Args:
        image:  PIL.Image input.
        masks:  list of np.ndarray binary masks.
        colors: list of [color_name, (R, G, B)] per mask.
        labels: optional list of str labels drawn at each mask's anchor
                (horizontal center, top edge).

    Returns:
        np.ndarray (H, W, 3) with masks overlaid.
    """
    overlay = np.array(image.copy()).astype(np.uint8)
    anchors = []
    for mask, color_cfg in zip(masks, colors):
        color = color_cfg[1]
        colored_mask = np.zeros_like(overlay, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = color[c]
        alpha = 0.3
        overlay[mask > 0] = (alpha * colored_mask[mask > 0] + (1 - alpha) * overlay[mask > 0]).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        if labels is not None:
            ys, xs = np.where(mask > 0)
            anchors.append((int(np.mean(xs)), int(np.min(ys))) if len(xs) > 0 else (0, 0))

    if labels is not None and anchors:
        pil_overlay = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_overlay)
        _draw_labels(draw, anchors, labels, [c[1] for c in colors], pil_overlay.size)
        overlay = np.array(pil_overlay)

    return overlay


def draw_boxes_on_image(image, boxes, colors, labels=None):
    """Draw colored bounding boxes on an image.

    Args:
        image:  PIL.Image input.
        boxes:  list of [x1, y1, x2, y2].
        colors: list of [color_name, (R, G, B)] per box.
        labels: optional list of str labels drawn at each box's top-center.

    Returns:
        np.ndarray (H, W, 3) with boxes drawn.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    anchors = []
    for box, color_cfg in zip(boxes, colors):
        x1, y1, x2, y2 = box
        color = color_cfg[1]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if labels is not None:
            anchors.append((int((x1 + x2) / 2), int(y1)))

    if labels is not None and anchors:
        _draw_labels(draw, anchors, labels, [c[1] for c in colors], overlay.size)

    return np.array(overlay)


def draw_points_on_image(image, points, colors, labels=None):
    """Draw colored circle markers at specified pixel coordinates.

    Point radius scales dynamically with image size (base: radius 6 at 640x480).

    Args:
        image:  PIL.Image input.
        points: list of [u, v] pixel coordinates.
        colors: list of [color_name, (R, G, B)] per point.
        labels: optional list of str labels drawn beside each point.

    Returns:
        np.ndarray (H, W, 3) with points drawn.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    img_w, img_h = overlay.size
    scale_factor = min(img_w / 640, img_h / 480)
    radius = max(2, int(6 * scale_factor))

    for pt, color_cfg in zip(points, colors):
        u, v = pt
        color = color_cfg[1]
        draw.ellipse([u - radius, v - radius, u + radius, v + radius], fill=color, outline=color)

    if labels is not None:
        anchors = [(pt[0], pt[1]) for pt in points]
        _draw_labels(draw, anchors, labels, [c[1] for c in colors], (img_w, img_h))

    return np.array(overlay)


# ─── Configuration ────────────────────────────────────────────────────────────

_ALLOWED_MARK_TYPES = ("point", "box")


@dataclass
class MarkConfig:
    """
    Configuration for visual marking behavior.

    Mask marks are disabled pipeline-wide; only ``point`` and ``box`` are used.
    """
    mark_types: List[str] = None          # e.g., ["box", "point"]
    type_weights: Optional[Dict[str, float]] = None   # e.g., {"point": 0.3, "box": 0.7}
    shuffle_colors: bool = False          # size.py shuffles colors; others don't


# ─── VisualMarker ─────────────────────────────────────────────────────────────

class VisualMarker:
    """
    Unified visual mark drawing for all annotation tasks.

    Manages color queue and dispatches to mask/box/point drawing functions.
    """

    def __init__(self, config: MarkConfig = None):
        self.config = config or MarkConfig()
        self.color_queue = list(COLOR_QUEUE_DEFAULT)

    def reset(self, shuffle: bool = None):
        """Reset color queue to default order, optionally shuffling."""
        self.color_queue = list(COLOR_QUEUE_DEFAULT)
        do_shuffle = shuffle if shuffle is not None else self.config.shuffle_colors
        if do_shuffle:
            rng().shuffle(self.color_queue)

    def pop_color(self) -> Tuple[str, tuple]:
        """Pop the next color from the queue."""
        name = self.color_queue.pop(0)
        return name, COLOR_MAP[name]

    @staticmethod
    def _allowed_types(types) -> List[str]:
        return [t for t in types if t in _ALLOWED_MARK_TYPES]

    def choose_mark_type(self) -> str:
        """
        Choose a mark type respecting configured weights (point/box only).
        """
        if self.config.type_weights:
            types = self._allowed_types(self.config.type_weights.keys())
            if not types:
                return "box"
            weights = [self.config.type_weights[t] for t in types]
            return rng().choices(types, weights=weights, k=1)[0]
        if self.config.mark_types:
            types = self._allowed_types(self.config.mark_types)
            if types:
                return rng().choice(types)
        return rng().choice(["point", "box"])

    @staticmethod
    def _extract(obj, view_idx=0):
        """Extract (tag, passthrough, bbox_2d, mask_pil) from a SceneNode or legacy tuple."""
        from .scene_graph import SceneNode
        if isinstance(obj, SceneNode):
            app = obj.view_appearances.get(view_idx)
            return (obj.tag, obj,
                    app.bbox_2d if app else None,
                    app.mask if app else None)
        # legacy tuple: (tag, passthrough, bbox_2d, mask_pil, ...)
        return (obj[0], obj[1],
                obj[2] if len(obj) > 2 else None,
                obj[3] if len(obj) > 3 else None)

    def plan_mark(
        self,
        objs: list = None,
        mark_type: str = None,
        view_idx: int = 0,
        labels: List[str] = None,
        points: list = None,
    ) -> Tuple[dict, list]:
        """
        Build mark_spec v2 without rendering (see mark_spec.py).

        Returns (mark_spec, marked_info) in the same shape as mark_objects().
        """
        from .mark_spec import plan_object_marks, plan_point_marks

        if points is not None:
            spec, point_slots = plan_point_marks(self, points, labels=labels)
            marked_info = [
                (f"{s.get('tag', 'point')}-({s['color_name']} point)", None)
                for s in point_slots
            ]
            return spec, marked_info
        spec, marked_info = plan_object_marks(
            self, objs, mark_type=mark_type, view_idx=view_idx, labels=labels,
        )
        return spec, marked_info

    def mark_objects(
        self,
        image: Image.Image,
        objs: list = None,
        mark_type: str = None,
        view_idx: int = 0,
        labels: List[str] = None,
        points: list = None,
        preprocess_row: dict = None,
    ) -> Tuple[dict, list]:
        """
        Plan mark_spec then render. Backward-compatible return shape.

        preprocess_row: optional parquet row for mask_ref resolution at render time.
        """
        from .mark_spec import render_mark

        spec, marked_info = self.plan_mark(
            objs=objs,
            mark_type=mark_type,
            view_idx=view_idx,
            labels=labels,
            points=points,
        )
        self._last_mark_spec = spec
        rendered = render_mark(image, spec, labels=labels, preprocess_row=preprocess_row)
        return rendered, marked_info

    @property
    def last_mark_spec(self) -> Optional[dict]:
        return getattr(self, "_last_mark_spec", None)
