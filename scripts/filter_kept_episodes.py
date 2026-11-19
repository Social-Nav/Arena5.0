#!/usr/bin/env python3
"""Copy a curated subset of human_interaction episodes into a new dataset tree.

kept_episodes.txt contains absolute paths to selected parquet files. The output
episodes are renumbered contiguously from zero, while instructions, task_index
links, frame_index, and timestamps are preserved.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


POSTPROCESS_DIR = Path(__file__).resolve().parents[1] / "arena_isaac" / "arena_isaac" / "arena_isaac" / "data_logging" / "postprocess"
if str(POSTPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_DIR))

from process_raw_to_dataset import (  # noqa: E402
    CAMERA_HEIGHT_CM,
    CAMERA_PITCH_DEG,
    episode_stats_entry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kept", default="src/human_interaction/kept_episodes.txt")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--output-name", default="human_interaction_filtered")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def infer_source_root(kept_path: Path) -> Path:
    for line in kept_path.read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        path = Path(candidate).expanduser()
        if "human_interaction" in path.parts:
            idx = path.parts.index("human_interaction")
            return Path(*path.parts[: idx + 1])
    raise SystemExit("Could not infer source root from kept_episodes.txt")


def load_kept_episodes(kept_path: Path, source_root: Path):
    kept: dict[str, set[int]] = defaultdict(set)
    for line in kept_path.read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        path = Path(candidate).expanduser()
        rel = path.relative_to(source_root)
        match = re.search(r"grscenes_(\d+)/data/chunk-\d+/episode_(\d+)\.parquet$", str(rel))
        if not match:
            raise SystemExit(f"Unexpected kept path format: {path}")
        kept[f"grscenes_{match.group(1)}"].add(int(match.group(2)))
    if not kept:
        raise SystemExit("kept_episodes.txt contains no episode paths")
    return kept


def old_parquet_path(world_dir: Path, episode: int) -> Path:
    return world_dir / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"


def new_parquet_path(world_dir: Path, episode: int) -> Path:
    return world_dir / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"


def copy_episode(
    source_world: Path,
    output_world: Path,
    old_episode: int,
    new_episode: int,
    index_offset: int,
    dry_run: bool,
):
    source_parquet = old_parquet_path(source_world, old_episode)
    if not source_parquet.is_file():
        raise FileNotFoundError(source_parquet)

    target_parquet = new_parquet_path(output_world, new_episode)
    if dry_run:
        print(target_parquet)
        for source_file in source_world.rglob(f"videos/**/episode_{old_episode:06d}_*"):
            rel = source_file.relative_to(source_world)
            target_rel = rel.with_name(rel.name.replace(f"episode_{old_episode:06d}", f"episode_{new_episode:06d}"))
            print(output_world / target_rel)
        return

    table = pq.read_table(source_parquet)
    n_rows = table.num_rows
    episode_col = pa.array([new_episode] * n_rows, type=pa.int64())
    index_col = pa.array(list(range(index_offset, index_offset + n_rows)), type=pa.int64())
    table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index", episode_col)
    table = table.set_column(table.schema.get_field_index("index"), "index", index_col)

    target_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target_parquet)

    for source_file in source_world.rglob(f"videos/**/episode_{old_episode:06d}_*"):
        rel = source_file.relative_to(source_world)
        target_rel = rel.with_name(rel.name.replace(f"episode_{old_episode:06d}", f"episode_{new_episode:06d}"))
        target_file = output_world / target_rel
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)


def rebuild_meta(
    source_world: Path,
    output_world: Path,
    mapping: dict[int, int],
    dry_run: bool,
):
    meta_src = source_world / "meta"
    meta_out = output_world / "meta"

    if dry_run:
        for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
            print(meta_out / name)
        return

    source_episodes = [
        json.loads(line)
        for line in (meta_src / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    old_episode_rows = {row["episode_index"]: row for row in source_episodes}

    task_indices: set[int] = set()
    for old_episode in mapping:
        table = pq.read_table(old_parquet_path(source_world, old_episode), columns=["task_index"])
        task_indices.update(table.column("task_index").to_pylist())

    source_tasks = [
        json.loads(line)
        for line in (meta_src / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept_tasks = [row for row in source_tasks if row.get("task_index") in task_indices]

    kept_episodes = []
    stats_rows = []
    index_offset = 0
    setting = f"{CAMERA_HEIGHT_CM}cm_{CAMERA_PITCH_DEG}deg"

    for old_episode in sorted(mapping):
        new_episode = mapping[old_episode]
        old_row = old_episode_rows[old_episode]
        new_row = dict(old_row)
        new_row["episode_index"] = new_episode
        kept_episodes.append(new_row)
        stats = episode_stats_entry(
            new_parquet_path(output_world, new_episode),
            new_episode,
            setting,
            index_offset,
        )
        stats_rows.append(stats)
        index_offset += old_row["length"]

    info = json.loads((meta_src / "info.json").read_text(encoding="utf-8"))
    info.update(
        {
            "total_episodes": len(kept_episodes),
            "total_frames": index_offset,
            "total_tasks": len(kept_tasks),
            "total_chunks": len({ep // 1000 for ep in mapping.values()}),
            "splits": {"train": f"0:{len(kept_episodes)}"},
        }
    )

    meta_out.mkdir(parents=True, exist_ok=True)
    (meta_out / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    (meta_out / "tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept_tasks),
        encoding="utf-8",
    )
    (meta_out / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept_episodes),
        encoding="utf-8",
    )
    (meta_out / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in stats_rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    kept_path = Path(args.kept).expanduser().resolve()
    if not kept_path.is_file():
        raise SystemExit(f"kept file not found: {kept_path}")

    source_root = (
        Path(args.source_root).expanduser().resolve()
        if args.source_root
        else infer_source_root(kept_path)
    )
    if not source_root.is_dir():
        raise SystemExit(f"source root not found: {source_root}")

    output_root = source_root.parent / args.output_name
    if output_root.exists() and not args.overwrite:
        raise SystemExit(f"output folder already exists: {output_root}; use --overwrite")
    if output_root.exists() and not args.dry_run:
        shutil.rmtree(output_root)

    kept = load_kept_episodes(kept_path, source_root)
    for world, episodes in sorted(kept.items()):
        source_world = source_root / "traj_data" / "grscenes" / world
        output_world = output_root / "traj_data" / "grscenes" / world
        sorted_episodes = sorted(episodes)
        mapping = {old: new for new, old in enumerate(sorted_episodes)}
        index_offset = 0
        for old_episode in sorted_episodes:
            old_row = next(
                json.loads(line)
                for line in (source_world / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["episode_index"] == old_episode
            )
            new_episode = mapping[old_episode]
            copy_episode(source_world, output_world, old_episode, new_episode, index_offset, args.dry_run)
            index_offset += old_row["length"]
        rebuild_meta(source_world, output_world, mapping, args.dry_run)

    print(
        f"{'Would write' if args.dry_run else 'Wrote'} "
        f"{sum(len(v) for v in kept.values())} episodes to {output_root}"
    )


if __name__ == "__main__":
    main()
