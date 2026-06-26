#!/usr/bin/env bash
set -e

python run.py --config config/annotation/demo_singleview_all.yaml --output_dir ./output/EmbodiedScan/arkitscenes/

python run.py --config config/aggregate/demo_aggregate_singleview.yaml --output_dir ./output/EmbodiedScan/arkitscenes/

python script/export_to_pangu_ml.py --export-dir output/EmbodiedScan/arkitscenes/base_pipeline_demo_aggregate_singleview/export_stage/ --output-root output/EmbodiedScan/arkitscenes/pangu_ml/singleview/ --num-workers 16 --intra-shard-workers 4
