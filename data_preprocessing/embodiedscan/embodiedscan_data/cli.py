import argparse
import logging
import sys

from embodiedscan_data.datasets import ALL_DATASETS


def cmd_extract(args):
    from embodiedscan_data.extract import extract_dataset

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        extract_dataset(
            dataset_name=ds,
            data_root=args.data_root,
            output_dir=args.output,
            workers=args.workers,
            max_scenes=args.max_scenes,
            arkit_asset_mode=args.arkit_asset_mode if ds == "arkitscenes" else None,
        )


def cmd_merge(args):
    import os
    from glob import glob
    from embodiedscan_data.merge import merge_to_scenes

    input_dir = args.input
    output_dir = args.output or input_dir

    jsonl_files = sorted(glob(os.path.join(input_dir, "*.jsonl")))
    jsonl_files = [f for f in jsonl_files if not f.endswith("_scenes.jsonl")]

    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return

    for filepath in jsonl_files:
        base = os.path.splitext(os.path.basename(filepath))[0]
        output_path = os.path.join(output_dir, f"{base}_scenes.jsonl")
        merge_to_scenes(filepath, output_path)
        print(f"Merged {filepath} -> {output_path}")


def cmd_export(args):
    from embodiedscan_data.export import export_to_parquet

    formats = []
    if args.format in ("both", "per_image"):
        formats.append("per_image")
    if args.format in ("both", "per_scene"):
        formats.append("per_scene")

    export_to_parquet(
        input_dir=args.input,
        output_dir=args.input,  # output alongside input
        batch_size=args.batch_size,
        formats=formats,
        hf_repo=args.hf_repo,
    )


def cmd_prepare_arkit(args):
    from embodiedscan_data.prepare_arkitscenes import prepare_arkitscenes

    prepare_arkitscenes(
        data_root=args.data_root,
        raw_root=args.raw_root,
        workers=args.workers,
        apply_sky_correction=not args.no_sky_correction,
        force=args.force,
        max_scenes=args.max_scenes,
        scene_filter=args.scene,
        only_local_raw=args.only_local_raw,
        sky_source=getattr(args, "sky_source", "traj"),
        sky_granularity=getattr(args, "sky_granularity", "frame"),
    )


def cmd_validate(args):
    from embodiedscan_data.validate import run_all

    passed = run_all(
        directory=args.input,
        data_root=args.data_root,
        sample_size=args.sample_size,
    )
    sys.exit(0 if passed else 1)


def main():
    parser = argparse.ArgumentParser(
        prog="embodiedscan-data",
        description="Unified data pipeline for EmbodiedScan datasets",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = sub.add_parser("extract", help="Extract per-image data from a dataset")
    p_extract.add_argument("--dataset", required=True, choices=ALL_DATASETS + ["all"])
    p_extract.add_argument("--data-root", required=True, help="Root data directory")
    p_extract.add_argument("--output", required=True, help="Output directory")
    p_extract.add_argument("--workers", type=int, default=24)
    p_extract.add_argument("--max-scenes", type=int, default=None, help="Limit scenes (for testing)")
    p_extract.add_argument(
        "--arkit-asset-mode",
        choices=("auto", "vga", "lowres"),
        default="auto",
        help="ARKitScenes only: auto=use prepared vga if present else lowres; "
        "vga=force prepared vga_wide; lowres=force lowres_wide (ignore prepared)",
    )
    p_extract.set_defaults(func=cmd_extract)

    # merge
    p_merge = sub.add_parser("merge", help="Merge per-image JSONL to per-scene")
    p_merge.add_argument("--input", required=True, help="Directory with per-image JSONL files")
    p_merge.add_argument("--output", default=None, help="Output directory (default: same as input)")
    p_merge.set_defaults(func=cmd_merge)

    # export
    p_export = sub.add_parser("export", help="Export JSONL to Parquet")
    p_export.add_argument("--input", required=True, help="Directory with JSONL files")
    p_export.add_argument("--format", choices=["per_image", "per_scene", "both"], default="both")
    p_export.add_argument("--batch-size", type=int, default=3000)
    p_export.add_argument("--hf-repo", default=None, help="HuggingFace repo ID for upload")
    p_export.set_defaults(func=cmd_export)

    # prepare-arkit
    p_arkit = sub.add_parser(
        "prepare-arkit",
        help="Prepare ARKitScenes vga_wide assets (sky correction + depth align)",
    )
    p_arkit.add_argument("--data-root", required=True, help="EmbodiedScan data root")
    p_arkit.add_argument(
        "--raw-root",
        default=None,
        help="Raw download root (default: <data-root>/arkitscenes_highres)",
    )
    p_arkit.add_argument("-j", "--workers", type=int, default=8, help="Parallel workers")
    p_arkit.add_argument(
        "--only-local-raw",
        action="store_true",
        help="Only scenes present under --raw-root",
    )
    p_arkit.add_argument(
        "-ss",
        "--sky-source",
        choices=("metadata", "traj", "auto"),
        default="traj",
        help="Scene-level sky only (-sg scene): metadata|traj|auto (default: traj)",
    )
    p_arkit.add_argument(
        "-sg",
        "--sky-granularity",
        choices=("scene", "frame"),
        default="frame",
        help="frame=per-frame traj; scene=one sky per video (default: frame)",
    )
    p_arkit.add_argument(
        "-ns",
        "--no-sky-correction",
        action="store_true",
        help="Pack vga without rotating to upright landscape",
    )
    p_arkit.add_argument("--force", action="store_true")
    p_arkit.add_argument("--max-scenes", type=int, default=None)
    p_arkit.add_argument(
        "--scene",
        action="append",
        default=None,
        help="sample_idx filter, e.g. arkitscenes/Training/40753679",
    )
    p_arkit.set_defaults(func=cmd_prepare_arkit)

    # validate
    p_validate = sub.add_parser("validate", help="Validate output data quality")
    p_validate.add_argument("--input", required=True, help="Directory with JSONL files")
    p_validate.add_argument("--data-root", default=None, help="Data root for path checks")
    p_validate.add_argument("--sample-size", type=int, default=100)
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Force-import all dataset configs to trigger registration
    import embodiedscan_data.datasets.scannet  # noqa: F401
    import embodiedscan_data.datasets.rscan3d  # noqa: F401
    import embodiedscan_data.datasets.matterport3d  # noqa: F401
    import embodiedscan_data.datasets.arkitscenes  # noqa: F401

    args.func(args)
