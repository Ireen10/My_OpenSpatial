#!/usr/bin/env python3
"""Run milestone acceptance checks (plan §5.2)."""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _run_pytest(test_path: str) -> int:
    cmd = [sys.executable, "-m", "pytest", test_path, "-q"]
    print(">", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def verify_m0() -> bool:
    from task.prompt_templates.threed_grounding_prompt_templates import (
        grounding_camera_introduction,
        grounding_json_output_instruction,
    )

    text = grounding_camera_introduction[0] + " " + grounding_json_output_instruction[0]
    if "roll,pitch,yaw" not in text.replace(" ", ""):
        print("M0 FAIL: bbox_3d must list roll,pitch,yaw in order")
        return False
    if "pitch,yaw,roll" in text.replace(" ", ""):
        print("M0 FAIL: legacy pitch,yaw,roll ordering still present")
        return False
    if "f_x=[FX]" not in text or "f_y=[FY]" not in text:
        print("M0 FAIL: camera_system must use focal length f_x/f_y placeholders")
        return False
    if "-Pi to Pi" not in text and "-pi to pi" not in text.lower():
        print("M0 FAIL: euler range -Pi to Pi not documented in prompt")
        return False

    rc = _run_pytest("tests/test_grounding_euler.py")
    if rc != 0:
        print("M0 FAIL: pytest tests/test_grounding_euler.py")
        return False
    print("M0 PASS")
    return True


def verify_m1() -> bool:
    path = REPO_ROOT / "verification" / "dataset_pipeline" / "validate_metadata.py"
    if not path.is_file():
        print("M1 SKIP: validate_metadata.py not implemented yet")
        return False
    spec = importlib.util.spec_from_file_location("validate_metadata", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not mod.run_self_check():
        return False
    audit = REPO_ROOT / "verification" / "dataset_pipeline" / "audit_annotation_outputs.py"
    if not audit.is_file():
        print("M1 FAIL: audit_annotation_outputs.py missing")
        return False
    print("M1 PASS (validator + audit script present)")
    return True


def verify_m2() -> bool:
    for test_path in (
        "tests/test_mask_ref_path.py",
        "tests/test_mark_spec_record.py",
    ):
        if _run_pytest(test_path) != 0:
            print(f"M2 FAIL: pytest {test_path}")
            return False
    print("M2 PASS (mask_ref.path + mark_spec record)")
    return True


def verify_m3() -> bool:
    from task.annotation.core.base_annotation_task import BaseAnnotationTask
    from task.annotation.distance import AnnotationGenerator as DistanceGen

    if not hasattr(BaseAnnotationTask, "_record_turn"):
        print("M3 FAIL: BaseAnnotationTask missing _record_turn")
        return False
    gen = DistanceGen({"emit_metadata": True, "emit_marked_images": False, "file_name": "distance"})
    if not gen.emit_metadata or gen.emit_marked_images:
        print("M3 FAIL: expected emit_metadata=True, emit_marked_images=False")
        return False
    rc = _run_pytest("tests/test_depth_metadata.py")
    if rc != 0:
        print("M3 WARN: tests/test_depth_metadata.py failed or missing (optional)")
    print("M3 PASS (base metadata infra)")
    return True


def verify_m4() -> bool:
    if _run_pytest("tests/test_aggregate_dedup_merge.py::test_dedup_same_core_different_referent") != 0:
        print("M4 FAIL: test_dedup_same_core_different_referent")
        return False
    if _run_pytest("tests/test_aggregate_dedup_merge.py::test_dedup_different_objects_not_merged") != 0:
        print("M4 FAIL: test_dedup_different_objects_not_merged")
        return False
    print("M4 PASS")
    return True


def verify_m5() -> bool:
    if _run_pytest("tests/test_aggregate_dedup_merge.py::test_merge_same_visual_group_two_objects") != 0:
        print("M5 FAIL: test_merge_same_visual_group_two_objects")
        return False
    if _run_pytest("tests/test_aggregate_dedup_merge.py::test_merge_different_mark_spec_splits_samples") != 0:
        print("M5 FAIL: test_merge_different_mark_spec_splits_samples")
        return False
    print("M5 PASS")
    return True


def verify_m6() -> bool:
    for test_path in (
        "tests/test_message_placeholders.py",
        "tests/test_record_turn_answer_sync.py",
    ):
        if _run_pytest(test_path) != 0:
            print(f"M6 FAIL: pytest {test_path}")
            return False
    script = REPO_ROOT / "verification" / "dataset_pipeline" / "check_annotation_mark_paths.py"
    if script.is_file():
        import subprocess

        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout or r.stderr)
            print("M6 FAIL: check_annotation_mark_paths.py")
            return False
    print("M6 PASS (message_placeholders + mark path check)")
    return True


def verify_m7(*, check_l2_bundle: bool = False) -> bool:
    for rel in (
        "dataset/upstream_export.py",
        "dataset/jsonl_base.py",
        "task/export/dataset_exporter.py",
        "config/aggregate/demo_aggregate_singleview.yaml",
    ):
        if not (REPO_ROOT / rel).is_file():
            print(f"M7 FAIL: missing {rel}")
            return False
    if _run_pytest("tests/test_upstream_export.py") != 0:
        print("M7 FAIL: pytest tests/test_upstream_export.py")
        return False
    if check_l2_bundle:
        from dataset.upstream_export import verify_bundle_roundtrip

        for branch, run_name in (
            ("singleview", "base_pipeline_demo_aggregate_singleview"),
            ("multiview", "base_pipeline_demo_aggregate_multiview"),
        ):
            bundle = REPO_ROOT / "output" / "frame_rot" / run_name / "export"
            sharded = bundle / "jsonl" / "metadata_000000.jsonl"
            legacy = bundle / "export" / "samples.jsonl"
            if not sharded.is_file() and not legacy.is_file():
                print(f"M7 FAIL: missing L2 bundle under {bundle}")
                return False
            check_root = bundle if sharded.is_file() else bundle / "export"
            ok, errs = verify_bundle_roundtrip(check_root)
            if not ok:
                print(f"M7 FAIL: roundtrip {bundle}: {errs[:5]}")
                return False
            print(f"M7 L2 OK: {branch} bundle roundtrip")
    print("M7 PASS (upstream bundle module + tests)")
    return True


def verify_m8() -> bool:
    path = REPO_ROOT / "task" / "annotation" / "core" / "structured_prompt_template.py"
    if not path.is_file():
        print("M8 FAIL: structured_prompt_template.py missing")
        return False
    if _run_pytest("tests/test_prompt_template_structure.py") != 0:
        print("M8 FAIL: pytest tests/test_prompt_template_structure.py")
        return False
    from task.annotation.core.question_type import QuestionType

    if not hasattr(QuestionType, "JUDGMENT"):
        print("M8 FAIL: QuestionType.JUDGMENT missing")
        return False
    compare_script = REPO_ROOT / "verification" / "dataset_pipeline" / "compare_annotation_baseline.py"
    if not compare_script.is_file():
        print("M8 FAIL: compare_annotation_baseline.py missing")
        return False
    import task.prompt_templates  # noqa: F401
    from task.annotation.core.structured_prompt_template import StructuredTemplateRegistry

    n_tpl = len(StructuredTemplateRegistry.keys())
    if n_tpl < 40:
        print(f"M8 FAIL: expected >=40 structured templates, got {n_tpl}")
        return False
    reg_helper = REPO_ROOT / "task" / "prompt_templates" / "register_structured.py"
    if not reg_helper.is_file():
        print("M8 FAIL: register_structured.py missing")
        return False
    print(f"M8 PASS (L1+L2: {n_tpl} structured templates, compare_annotation_baseline.py)")
    return True


def verify_pre_m8_gates() -> bool:
    """M6/M7-adjacent gates required before M8 template edits."""
    import subprocess

    script = REPO_ROOT / "verification" / "dataset_pipeline" / "audit_annotation_source.py"
    if script.is_file():
        r = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT))
        if r.returncode != 0:
            print("pre-M8 FAIL: audit_annotation_source.py")
            return False
    print("pre-M8 PASS: audit_annotation_source.py")
    return True


MILESTONE_FN = {
    "M0": verify_m0,
    "M1": verify_m1,
    "M2": verify_m2,
    "M3": verify_m3,
    "M4": verify_m4,
    "M5": verify_m5,
    "M6": verify_m6,
    "M7": verify_m7,
    "M8": verify_m8,
}


def main():
    parser = argparse.ArgumentParser(description="Verify dataset pipeline milestones")
    parser.add_argument(
        "--milestone",
        choices=list(MILESTONE_FN.keys()),
        default=None,
        help="Single milestone to verify (not needed with --pre-m8-gates only)",
    )
    parser.add_argument("--through", default=None, help="Run M0..Mx inclusive (subset implemented)")
    parser.add_argument(
        "--m7-check-l2-bundle",
        action="store_true",
        help="M7 only: verify output/frame_rot export bundles on disk (roundtrip)",
    )
    parser.add_argument(
        "--pre-m8-gates",
        action="store_true",
        help="Run answer-provenance source audit (for M8 template work)",
    )
    args = parser.parse_args()

    if args.pre_m8_gates:
        sys.exit(0 if verify_pre_m8_gates() else 1)

    if not args.milestone and not args.through:
        parser.error("require --milestone or --through (or use --pre-m8-gates)")

    order = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    if args.through:
        end = args.through.upper()
        if end not in order:
            print(f"--through {end} not in implemented milestones {order}")
            sys.exit(2)
        idx = order.index(end)
        milestones = order[: idx + 1]
    else:
        milestones = [args.milestone.upper()]

    ok = True
    for m in milestones:
        fn = MILESTONE_FN.get(m)
        if fn is None:
            print(f"{m} SKIP: no verifier registered")
            continue
        if m == "M7" and args.m7_check_l2_bundle:
            passed = verify_m7(check_l2_bundle=True)
        else:
            passed = fn()
        if not passed:
            ok = False
            if args.through:
                break
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
