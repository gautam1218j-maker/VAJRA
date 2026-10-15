"""Generate four reproducible days of synthetic VAJRA training data."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import db
from data_pipeline import (
    GRID_H,
    GRID_W,
    OUTPUT_STEPS,
    INPUT_STEPS,
    SIMULATION_SEED,
    TIME_STEP_MIN,
    SyntheticDataSimulator,
)

DEFAULT_START = datetime(2024, 6, 1, tzinfo=timezone.utc)
FRAMES_PER_DAY = 24 * 60 // TIME_STEP_MIN
SPLITS = (
    ("train", 0, 2 * FRAMES_PER_DAY),
    ("val", 2 * FRAMES_PER_DAY, 3 * FRAMES_PER_DAY),
    ("test", 3 * FRAMES_PER_DAY, 4 * FRAMES_PER_DAY),
)
SOURCES = ("radar", "satellite", "lightning", "nwp")


def generate_dataset(
    output_dir: str | Path = "data/starter_dataset",
    start: datetime = DEFAULT_START,
    seed: int = SIMULATION_SEED,
    force: bool = False,
) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{output_dir} is not empty; pass --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = start.astimezone(timezone.utc)
    db_path = output_dir / "vajra.sqlite3"
    db.init_db(f"sqlite:///{db_path.as_posix()}", output_dir)

    simulator = SyntheticDataSimulator(seed=seed)
    frame_rows: list[dict] = []
    event_count = 0
    frame_count = 4 * FRAMES_PER_DAY
    for index in range(frame_count):
        timestamp = start + timedelta(minutes=index * TIME_STEP_MIN)
        frame, events = simulator.frame_at(timestamp)
        grids = {
            "radar": frame.radar_dbz,
            "satellite": frame.ir_temp,
            "lightning": frame.lightning_density,
            "nwp": np.stack((frame.cape, frame.pwat)).astype(np.float32),
        }
        records = {}
        for source in SOURCES:
            records[source] = db.insert_frame(timestamp, source, grids[source], commit=False)
        frame_rows.append(records)
        event_count += db.insert_lightning_events(events, commit=False)
        if (index + 1) % 18 == 0:
            db.commit_database()
            print(f"Generated {index + 1}/{frame_count} timestamps")
    db.commit_database()

    sequence_id = 0
    sequence_counts = {}
    window_steps = INPUT_STEPS + OUTPUT_STEPS
    for split, split_start, split_end in SPLITS:
        count = split_end - split_start - window_steps + 1
        sequence_counts[split] = count
        for local_start in range(split_start, split_end - window_steps + 1):
            sequence_id += 1
            for offset in range(window_steps):
                role = "input" if offset < INPUT_STEPS else "target"
                frame_index = local_start + offset
                for source in SOURCES:
                    db.insert_sequence(
                        sequence_id,
                        frame_rows[frame_index][source].id,
                        role,
                        split,
                        commit=False,
                    )
            if sequence_id % 50 == 0:
                db.commit_database()
        db.commit_database()

    manifest = {
        "project": "VAJRA synthetic starter dataset",
        "start_utc": start.isoformat(),
        "end_utc_exclusive": (start + timedelta(days=4)).isoformat(),
        "cadence_minutes": TIME_STEP_MIN,
        "logical_timestamps": frame_count,
        "grid_shape": [GRID_H, GRID_W],
        "seed": seed,
        "sequence_window": {"input_steps": INPUT_STEPS, "target_steps": OUTPUT_STEPS},
        "split_dates_utc": {
            "train": [start.date().isoformat(), (start + timedelta(days=2)).date().isoformat()],
            "val": [(start + timedelta(days=2)).date().isoformat(), (start + timedelta(days=3)).date().isoformat()],
            "test": [(start + timedelta(days=3)).date().isoformat(), (start + timedelta(days=4)).date().isoformat()],
        },
        "sequence_counts": sequence_counts,
        "lightning_event_count": event_count,
        "database": "vajra.sqlite3",
        "grid_directory": "grids/",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    db._engine.dispose()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/starter_dataset")
    parser.add_argument("--start-date", default=DEFAULT_START.date().isoformat())
    parser.add_argument("--seed", type=int, default=SIMULATION_SEED)
    parser.add_argument("--force", action="store_true", help="replace a non-empty output directory")
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
    manifest = generate_dataset(args.output_dir, start=start, seed=args.seed, force=args.force)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
