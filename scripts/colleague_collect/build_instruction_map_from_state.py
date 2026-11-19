#!/usr/bin/env python3
"""Build a per-episode instruction map from the completed-scenario state file.

Scenario names use the form ``<interaction_type>_<number>``, for example
``robot_overtaking_008``. Completed scenarios are appended to the state file in
episode order, so the line index is the episode index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


_INTERACTION_ALIASES = {
    "scenario": "perpendicular_traffic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-state")
    parser.add_argument("--episode-map")
    parser.add_argument("--instruction-config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = Path(args.scenario_state) if args.scenario_state else None
    episode_map_path = Path(args.episode_map) if args.episode_map else None
    config_path = Path(args.instruction_config)
    output_path = Path(args.output)

    if episode_map_path is None and (state_path is None or not state_path.is_file()):
        raise SystemExit(f"scenario state file not found: {state_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    instructions = config.get("instructions")
    if not isinstance(instructions, dict):
        raise SystemExit(f"{config_path} must contain an 'instructions' mapping")

    episode_instructions = {}
    if episode_map_path is not None:
        if not episode_map_path.is_file():
            raise SystemExit(f"episode map file not found: {episode_map_path}")
        for line in episode_map_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            scenario_name, episode_index = line.split("\t", 1)
            episode_index = int(episode_index)
            interaction_type = scenario_name.rsplit("_", 1)[0]
            interaction_type = _INTERACTION_ALIASES.get(interaction_type, interaction_type)
            instruction = instructions.get(interaction_type)
            if instruction:
                episode_instructions[str(episode_index)] = instruction
    else:
        for episode_index, line in enumerate(
            state_path.read_text(encoding="utf-8").splitlines()
        ):
            scenario_name = line.strip()
            if not scenario_name:
                continue
            interaction_type = scenario_name.rsplit("_", 1)[0]
            interaction_type = _INTERACTION_ALIASES.get(interaction_type, interaction_type)
            instruction = instructions.get(interaction_type)
            if instruction:
                episode_instructions[str(episode_index)] = instruction

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(episode_instructions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(episode_instructions)} episode instruction(s) to {output_path}"
    )


if __name__ == "__main__":
    main()
