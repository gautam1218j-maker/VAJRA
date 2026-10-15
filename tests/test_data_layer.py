from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
import torch
from sqlalchemy import select

import db
from data_pipeline import (
    GRID_H,
    GRID_W,
    INPUT_STEPS,
    OUTPUT_STEPS,
    SyntheticDataSimulator,
    TIME_STEP_MIN,
    normalize,
)
from normalization import load_normalization_stats

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def configured_db():
    root = PROJECT_DIR / "data" / f"_test_db_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    db.init_db(f"sqlite:///{(root / 'test.sqlite3').as_posix()}", root)
    try:
        yield root
    finally:
        db.commit_database()
        db._engine.dispose()
        shutil.rmtree(root, ignore_errors=True)


def test_simulated_grids_are_finite_and_in_range():
    simulator = SyntheticDataSimulator(seed=13)
    timestamp = datetime(2024, 6, 1, tzinfo=timezone.utc)
    frame, _ = simulator.frame_at(timestamp)
    arrays = (frame.radar_dbz, frame.ir_temp, frame.lightning_density, frame.cape, frame.pwat)
    assert all(array.shape == (GRID_H, GRID_W) for array in arrays)
    assert all(np.isfinite(array).all() for array in arrays)
    assert frame.radar_dbz.dtype == np.float32
    assert frame.ir_temp.dtype == np.float32
    assert frame.cape.dtype == np.float32
    assert frame.pwat.dtype == np.float32
    assert np.all((-30 <= frame.radar_dbz) & (frame.radar_dbz <= 75))
    assert np.all((180 <= frame.ir_temp) & (frame.ir_temp <= 320))
    assert np.issubdtype(frame.lightning_density.dtype, np.integer)
    assert np.all(frame.lightning_density >= 0)
    assert np.all(frame.cape >= 0)


def test_storm_fields_are_seeded_and_temporally_continuous():
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    first = SyntheticDataSimulator(seed=27)
    duplicate = SyntheticDataSimulator(seed=27)
    radar0 = first.frame_at(start)[0].radar_dbz
    radar1 = first.frame_at(start + timedelta(minutes=TIME_STEP_MIN))[0].radar_dbz
    radar_random = first.frame_at(start + timedelta(hours=5))[0].radar_dbz
    radar_repeat = duplicate.frame_at(start)[0].radar_dbz
    adjacent = np.corrcoef(radar0.ravel(), radar1.ravel())[0, 1]
    distant = np.corrcoef(radar0.ravel(), radar_random.ravel())[0, 1]
    assert np.array_equal(radar0, radar_repeat)
    assert adjacent > distant


def test_individual_storm_cell_advects_by_its_velocity_each_step():
    simulator = SyntheticDataSimulator(seed=41)
    simulator._spawn_cell()
    cell = simulator.cells[0]
    speed = np.hypot(cell.velocity_y, cell.velocity_x)
    assert 2.0 <= speed <= 4.0

    if abs(cell.velocity_y) >= abs(cell.velocity_x):
        cell.y = GRID_H - 1 if cell.velocity_y > 0 else 0
        cell.x = GRID_W / 2
    else:
        cell.y = GRID_H / 2
        cell.x = GRID_W - 1 if cell.velocity_x > 0 else 0

    for step in (1, 2):
        previous_y, previous_x = cell.y, cell.x
        simulator._make_frame(step)
        assert cell.y - previous_y == pytest.approx(cell.velocity_y)
        assert cell.x - previous_x == pytest.approx(cell.velocity_x)
        assert np.hypot(cell.y - previous_y, cell.x - previous_x) == pytest.approx(speed)


def test_normalize_uses_shared_archive_statistics():
    means, stds = load_normalization_stats()
    values = np.stack((means, means + stds))[:, None, None, :]
    normalized = normalize(values)
    np.testing.assert_allclose(normalized[0, 0, 0], np.zeros(5), atol=1e-6)
    np.testing.assert_allclose(normalized[1, 0, 0], np.ones(5), atol=1e-6)


def test_nwp_is_held_between_three_hour_cycles():
    simulator = SyntheticDataSimulator(seed=31)
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    cape0, pwat0 = simulator.frame_at(start)[0].cape, simulator.frame_at(start)[0].pwat
    cape_later, pwat_later = simulator.frame_at(start + timedelta(hours=2, minutes=50))[0].cape, simulator.frame_at(start + timedelta(hours=2, minutes=50))[0].pwat
    cape_next, pwat_next = simulator.frame_at(start + timedelta(hours=3))[0].cape, simulator.frame_at(start + timedelta(hours=3))[0].pwat
    assert np.array_equal(cape0, cape_later)
    assert np.array_equal(pwat0, pwat_later)
    assert not np.array_equal(cape0, cape_next)
    assert not np.array_equal(pwat0, pwat_next)


def test_lightning_is_concentrated_over_strong_radar():
    simulator = SyntheticDataSimulator(seed=26072)
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    lightning_dbz = []
    for index in range(48):
        frame, _ = simulator.frame_at(start + timedelta(minutes=index * TIME_STEP_MIN))
        lightning_dbz.extend(frame.radar_dbz[frame.lightning_density > 0].tolist())
    assert lightning_dbz
    assert np.mean(np.asarray(lightning_dbz) > 20) > 0.70


def test_database_grid_and_lightning_round_trip(configured_db):
    timestamp = datetime(2024, 6, 1, tzinfo=timezone.utc)
    grid = np.arange(16, dtype=np.float32).reshape(4, 4)
    frame = db.insert_frame(timestamp, "radar", grid)
    events = [{
        "timestamp": timestamp,
        "lat": 18.5,
        "lon": 73.2,
        "grid_y": 12,
        "grid_x": 21,
        "strike_type": "CG",
    }]
    assert db.insert_lightning_events(events) == 1
    rows = db.query_by_timerange(timestamp, timestamp + timedelta(minutes=10), "radar")
    assert len(rows) == 1
    assert rows[0].id == frame.id
    assert np.array_equal(np.load(db.PROJECT_DIR / rows[0].grid_path), grid)
    with db._session_factory() as session:
        stored = session.scalars(select(db.LightningEvent)).one()
        assert (stored.lat, stored.lon, stored.grid_y, stored.grid_x, stored.strike_type) == (18.5, 73.2, 12, 21, "CG")
        assert stored.timestamp == timestamp.replace(tzinfo=None)


def test_database_sequence_has_contiguous_timestamps_and_loads_batch(configured_db):
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    sources = ("radar", "satellite", "lightning", "nwp")
    for index in range(INPUT_STEPS + OUTPUT_STEPS):
        timestamp = start + timedelta(minutes=index * TIME_STEP_MIN)
        role = "input" if index < INPUT_STEPS else "target"
        for source in sources:
            if source == "nwp":
                grid = np.stack((np.full((4, 4), 900), np.full((4, 4), 40))).astype(np.float32)
            elif source == "lightning":
                grid = np.zeros((4, 4), dtype=np.int32)
            elif source == "satellite":
                grid = np.full((4, 4), 260, dtype=np.float32)
            else:
                grid = np.full((4, 4), 15 + index, dtype=np.float32)
            frame = db.insert_frame(timestamp, source, grid, commit=False)
            db.insert_sequence(1, frame.id, role, "train", commit=False)
    db.commit_database()

    records = db.get_sequence(1)
    timestamps = sorted({row["timestamp"] for row in records})
    assert len(timestamps) == INPUT_STEPS + OUTPUT_STEPS
    assert all(right - left == timedelta(minutes=TIME_STEP_MIN) for left, right in zip(timestamps, timestamps[1:]))
    dataset = db.DBSequenceDataset(split="train")
    x, y_radar, y_lightning = dataset[0]
    means, stds = load_normalization_stats()
    raw_input = np.array([15, 260, 0, 900, 40], dtype=np.float32)
    np.testing.assert_allclose(x[0, :, 0, 0].numpy(), (raw_input - means) / stds)
    assert x.shape == (INPUT_STEPS, 5, 4, 4)
    assert y_radar.shape == (OUTPUT_STEPS, 4, 4)
    assert y_lightning.shape == (OUTPUT_STEPS, 4, 4)
    assert x.dtype == torch.float32
