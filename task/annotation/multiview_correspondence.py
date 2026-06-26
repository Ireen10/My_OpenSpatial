"""
Multiview point correspondence annotation task.

Given two views of the same scene, finds a 3D point visible in both views,
marks a query point on View 1, and asks the model to identify the
corresponding point among labeled candidates on View 2.

Sub-tasks:
    point_correspondence_oe  - open-ended: sentence or free (full-sentence answers with [L])
    point_correspondence_mcq - MCQ: direct/sentence/free; direct uses [T], sentence uses [L] then [E]

Algorithm (vectorized Open3D):
    1. Use _find_overlapping_views to find two diverse views sharing a common object.
    2. Backproject valid depth pixels to 3D world coordinates for both views.
    3. Use Open3D's compute_point_cloud_distance for vectorized nearest-neighbor
       overlap detection within `overlap_dist` threshold.
    4. Randomly select one overlapping point, project it back to 2D in both views.
    5. Generate 3 distractor points on View 2 (at least `min_distractor_dist`
       pixels away from the ground-truth point).
    6. Shuffle GT + distractors, assign labels (A/B/C/D or 1/2/3/4).

Templates used:
    multiview_correspondence.point2point[_{num}].oe.{sentence|free} - OE; same answer pool
    multiview_correspondence.point2point[_{num}].mcq.{direct|sentence|free} - MCQ; options via [O]

Configurable parameters (via YAML args):
    overlap_dist          - 3D distance threshold (meters) for two points to be
                            considered overlapping (default: 0.003)
    min_overlap_points    - minimum number of overlapping 3D points required to
                            accept a view pair (default: 10)
    boundary_margin       - pixel margin from image edge; points too close to
                            the border are rejected (default: 10)
    min_distractor_dist   - minimum pixel distance between each distractor and
                            the ground-truth point on View 2 (default: 20)
"""

from task.annotation.core.thread_rng import rng
import numpy as np
import open3d as o3d
from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.mark_spec import assemble_per_view_mark_spec
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Multiview Correspondence"
    SUB_TASKS = {
        "point_correspondence_oe":  {"default": 1, "handler": "_generate_point_correspondence_oe"},
        "point_correspondence_mcq": {"default": 1, "handler": "_generate_point_correspondence_mcq"},
    }
    _OE_INSTRUCTION_MODES = ("sentence", "free")
    _MCQ_INSTRUCTION_MODES = ("direct", "sentence", "free")

    def __init__(self, args):
        super().__init__(args)
        self.overlap_dist = args.get("overlap_dist", 0.003)
        self.min_overlap_points = args.get("min_overlap_points", 10)
        self.boundary_margin = args.get("boundary_margin", 10)
        self.min_distractor_dist = args.get("min_distractor_dist", 20)

    # --- Prompt Function ---

    @staticmethod
    def _correspondence_option_markers(use_numeric_labels: bool):
        """Option labels shared by MCQ options and OE answers (letter + point id)."""
        if use_numeric_labels:
            return [f"{letter}. Point {n}" for letter, n in zip("ABCD", "1234")]
        return [f"{letter}. Point {letter}" for letter in "ABCD"]

    def point_correspondence_prompt_func(self, question_color, candidate_color, gt_answer, question_type=QuestionType.OPEN_ENDED):
        """Build a point correspondence QA string.

        MCQ uses register_mcq templates with [O] (options in question, type MCQ).
        OE uses register_oe templates without [O]; no question_instruction on OE.

        Returns:
            (prompt, template_id)
        """
        is_num = gt_answer in ["1", "2", "3", "4"]
        base = (
            "multiview_correspondence.point2point_num"
            if is_num
            else "multiview_correspondence.point2point"
        )
        markers = self._correspondence_option_markers(is_num)
        label_order = ["1", "2", "3", "4"] if is_num else ["A", "B", "C", "D"]
        idx = label_order.index(gt_answer)
        full_option = markers[idx]
        option_letter = "ABCD"[idx]
        point_label = f"Point {gt_answer}"
        answer_slots = {
            "L": point_label,
            "E": option_letter,
            "T": full_option,
        }

        from .metric_gating import pick_instruction_mode

        if question_type == QuestionType.MCQ:
            mode = pick_instruction_mode(self._MCQ_INSTRUCTION_MODES)
            tpl_name = f"{base}.mcq.{mode}"
            shared = {
                "A": question_color,
                "B": candidate_color,
                "O": "\nOptions: " + " ".join(markers),
            }
            prompt = self.render_structured_prompt(
                tpl_name, shared=shared, a_args=answer_slots,
            )
        else:
            mode = pick_instruction_mode(self._OE_INSTRUCTION_MODES)
            tpl_name = f"{base}.oe.{mode}"
            prompt = self.render_structured_prompt(
                tpl_name,
                shared={"A": question_color, "B": candidate_color},
                a_args={"L": point_label},
            )
        return prompt, tpl_name

    # --- Point Correspondence Finder ---

    def _find_point_correspondence(self, graph):
        """Find a corresponding 3D point visible in two overlapping views.

        Returns:
            (meta_data, True) on success, where meta_data contains:
                image:      [PIL.Image, PIL.Image] - the two view images
                view_idx:   [int, int]             - view indices in the SceneGraph
                point:      [pt1_uv, pt2_uv, distractor_uvs]
                            pt1_uv:         [u, v] query point on View 1
                            pt2_uv:         [u, v] ground-truth point on View 2
                            distractor_uvs: [[u,v], ...] x 3 distractor points on View 2
            (None, False) on failure.
        """
        # Step 1: get two diverse views sharing a common object
        node, views = self._find_overlapping_views(graph, num_views=2)
        if node is None:
            return None, False

        view1_idx, view2_idx = views
        view1 = graph.views[view1_idx]
        view2 = graph.views[view2_idx]

        intrinsic1 = view1.intrinsic
        intrinsic2 = view2.intrinsic
        depth_map1 = view1.depth_map
        depth_map2 = view2.depth_map
        h1, w1 = depth_map1.shape
        h2, w2 = depth_map2.shape
        img_dim1 = (w1, h1)
        img_dim2 = (w2, h2)

        # Step 2: backproject valid depth pixels to 3D world coordinates.
        # Each view may have its own (W, H) after per-frame sky rotation (e.g. 640x480 vs 480x640).
        points_3d_1 = self.backproject_2d_to_3d(view1.pose, depth_map1, img_dim1, intrinsic1)
        points_3d_2 = self.backproject_2d_to_3d(view2.pose, depth_map2, img_dim2, intrinsic2)

        valid1 = depth_map1.ravel() > 0
        valid2 = depth_map2.ravel() > 0
        pts1_valid = points_3d_1[valid1]
        pts2_valid = points_3d_2[valid2]

        min_pts = self.min_overlap_points
        if len(pts1_valid) < min_pts or len(pts2_valid) < min_pts:
            return None, False

        # Step 3: compute nearest-neighbor distances (vectorized via Open3D)
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()

        # Subsample both views - full-depth clouds can be 300k+ pts and stall workers.
        max_pts = 50000
        if len(pts1_valid) > max_pts:
            query_idx = rng().sample(range(len(pts1_valid)), max_pts)
            pts1_query = pts1_valid[query_idx]
        else:
            pts1_query = pts1_valid
        if len(pts2_valid) > max_pts:
            target_idx = rng().sample(range(len(pts2_valid)), max_pts)
            pts2_target = pts2_valid[target_idx]
        else:
            pts2_target = pts2_valid

        from utils.point_cloud_utils import point_cloud_nn_distances

        pcd1.points = o3d.utility.Vector3dVector(pts1_query)
        pcd2.points = o3d.utility.Vector3dVector(pts2_target)

        nn_dists = point_cloud_nn_distances(pcd1, pcd2)
        overlap_indices = np.where(nn_dists < self.overlap_dist)[0]

        if len(overlap_indices) < min_pts:
            return None, False

        # Step 4: pick a random overlapping point, project back to 2D in both views
        sel_local = rng().choice(overlap_indices)
        sel_pt = pts1_query[sel_local]
        # Single-point nearest neighbor via numpy broadcast (faster than building KDTree)
        corr_idx = np.argmin(np.sum((pts2_valid - sel_pt) ** 2, axis=1))
        corr_pt = pts2_valid[corr_idx]

        inv_pose1 = np.linalg.inv(view1.pose)
        inv_pose2 = np.linalg.inv(view2.pose)
        pt1_uv = self.project_3d_to_2d(inv_pose1, sel_pt.reshape(1, 3), intrinsic1)[0]
        pt2_uv = self.project_3d_to_2d(inv_pose2, corr_pt.reshape(1, 3), intrinsic2)[0]

        # Reject if either projected point is too close to the image boundary
        margin = self.boundary_margin
        if not (margin <= pt1_uv[0] <= w1 - margin and margin <= pt1_uv[1] <= h1 - margin):
            return None, False
        if not (margin <= pt2_uv[0] <= w2 - margin and margin <= pt2_uv[1] <= h2 - margin):
            return None, False

        # Step 5: generate 3 distractor points on View 2 (far enough from GT)
        min_dist_sq = self.min_distractor_dist ** 2
        uv_candidates = []
        for _ in range(100):
            u = rng().randint(margin, w2 - margin - 1)
            v = rng().randint(margin, h2 - margin - 1)
            if (u - pt2_uv[0]) ** 2 + (v - pt2_uv[1]) ** 2 < min_dist_sq:
                continue
            uv_candidates.append([u, v])
            if len(uv_candidates) >= 3:
                break
        if len(uv_candidates) < 3:
            return None, False

        meta_data = {
            "image": [view1.image, view2.image],
            "view_idx": [view1_idx, view2_idx],
            "point": [pt1_uv.astype(int).tolist(), pt2_uv.astype(int).tolist(), uv_candidates],
        }
        return meta_data, True

    # --- Visual Marking ---

    def _draw_candidate_points(self, image1, image2, point1_uv, point2_uv, candidates_uv, meta):
        """Draw query point on View 1 and labeled candidate points on View 2.

        View 1 gets a single unlabeled point (the query).
        View 2 gets 4 labeled points (1 GT + 3 distractors) in shuffled order.

        Args:
            image1, image2: PIL images for the two views.
            point1_uv:      [u, v] query point on View 1.
            point2_uv:      [u, v] ground-truth corresponding point on View 2.
            candidates_uv:  [[u, v], ...] x 3 distractor points on View 2.

        Returns:
            (processed_image1, processed_image2,
             color_name1, color_name2, gt_answer)
            where gt_answer is the label assigned to the GT point after shuffling.
        """
        # Plan marks per view; merge so metadata retains query + candidate points (M2/M3).
        self.marker.reset(shuffle=True)
        spec1, color_name1 = self.marker.plan_mark(points=[point1_uv])
        self.marker._last_mark_spec = spec1

        all_points = [point2_uv] + candidates_uv
        indices = list(range(len(all_points)))
        rng().shuffle(indices)
        labels = ["1", "2", "3", "4"] if rng().random() < 0.3 else ["A", "B", "C", "D"]
        shuffled = [all_points[i] for i in indices]
        gt_answer = labels[indices.index(0)]

        spec2, color_name2 = self.marker.plan_mark(points=shuffled, labels=labels)
        row = getattr(self._thread_local, "preprocess_row", None)
        refs = self._qa_image_refs(row, meta)
        merged = assemble_per_view_mark_spec([
            {
                "view_index": 0,
                "image_ref": refs[0] if refs else None,
                "mark_kinds": spec1.get("mark_kinds", []),
                "slots": spec1.get("slots", []),
            },
            {
                "view_index": 1,
                "image_ref": refs[1] if len(refs) > 1 else None,
                "mark_kinds": spec2.get("mark_kinds", []),
                "slots": spec2.get("slots", []),
            },
        ])
        if merged:
            self.marker._last_mark_spec = merged

        if self.emit_marked_images:
            from .core.mark_spec import render_mark
            processed_image1 = render_mark(image1, merged, view_index=0)
            processed_image2 = render_mark(image2, merged, view_index=1, labels=labels)
        else:
            processed_image1 = {"bytes": convert_pil_to_bytes(image1)}
            processed_image2 = {"bytes": convert_pil_to_bytes(image2)}

        return processed_image1, processed_image2, color_name1, color_name2, gt_answer

    # --- Handlers ---

    def _build_correspondence(self, graph, question_type):
        """Shared pipeline for both OE and MCQ handlers.

        Retries up to 5 times to find a valid point correspondence,
        then draws visual marks and generates the prompt.
        """
        for _ in range(5):
            meta, flag = self._find_point_correspondence(graph)
            if flag:
                break
        if not flag:
            return None

        point1, point2, uv_candidates = meta["point"]
        image1, image2 = meta["image"]

        self.marker.reset(shuffle=True)
        img1, img2, color1, color2, gt_answer = self._draw_candidate_points(
            image1, image2, point1, point2, uv_candidates, meta,
        )

        key_points = [point1, point2] + list(uv_candidates or [])
        if not self.register_semantic_candidate(
            "multiview_correspondence.point2point",
            "MCQ" if question_type == QuestionType.MCQ else "OE",
            key_points,
            gt_answer,
        ):
            return None
        prompt, tpl = self.point_correspondence_prompt_func(color1, color2, gt_answer, question_type)
        if prompt is None:
            return None
        qtype = QuestionType.MCQ if question_type == QuestionType.MCQ else QuestionType.OPEN_ENDED
        sub = "point_correspondence_mcq" if qtype == QuestionType.MCQ else "point_correspondence_oe"
        self._record_turn(
            sub, tpl, prompt, qtype,
            mark_spec=self.marker.last_mark_spec,
            image_placeholder_count=2,
            view_indices=meta.get("view_idx"),
        )
        return prompt, [img1, img2], qtype

    def _generate_point_correspondence_oe(self, graph):
        """Generate an open-ended point correspondence QA."""
        return self._build_correspondence(graph, QuestionType.OPEN_ENDED)

    def _generate_point_correspondence_mcq(self, graph):
        """Generate an MCQ point correspondence QA."""
        return self._build_correspondence(graph, QuestionType.MCQ)
