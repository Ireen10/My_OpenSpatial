import json
import logging
import os
from collections import defaultdict
from typing import Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Fields that stay scalar in per-scene output (not aggregated into lists)
SCENE_LEVEL_FIELDS = {"dataset", "scene_id", "is_metric_depth"}


def merge_to_scenes(input_path: str, output_path: Optional[str] = None) -> str:
    """Merge per-image JSONL into per-scene JSONL.

    Groups records by scene_id and aggregates all per-image fields into lists.
    Adds num_images count.

    Args:
        input_path: Path to per-image JSONL file
        output_path: Path to output per-scene JSONL. Defaults to <input>_scenes.jsonl

    Returns:
        Path to the output file
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_scenes{ext}"

    grouped = defaultdict(list)
    bad_lines = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(
            tqdm(
                f,
                desc=f"Reading {os.path.basename(input_path)}",
                unit="line",
            ),
            1,
        ):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at line %d in %s", line_num, input_path)
                bad_lines += 1
                continue

            scene_id = record.get("scene_id")
            if scene_id is None:
                logger.warning("Missing scene_id at line %d in %s", line_num, input_path)
                bad_lines += 1
                continue

            grouped[scene_id].append(record)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for scene_id, records in tqdm(
            grouped.items(),
            total=len(grouped),
            desc=f"Writing {os.path.basename(output_path)}",
            unit="scene",
        ):
            scene_record = {
                "dataset": records[0].get("dataset", ""),
                "scene_id": scene_id,
                "num_images": len(records),
                "is_metric_depth": records[0].get("is_metric_depth", True),
            }
            # Aggregate all non-scene-level fields into lists
            for key in records[0]:
                if key not in SCENE_LEVEL_FIELDS:
                    scene_record[key] = [r.get(key) for r in records]

            f.write(json.dumps(scene_record, ensure_ascii=False) + "\n")

    if bad_lines > 0:
        logger.info("Skipped %d bad lines in %s", bad_lines, input_path)

    logger.info("Merged %d scenes from %s -> %s", len(grouped), input_path, output_path)
    return output_path
