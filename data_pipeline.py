"""
VAJRA Nowcasting — Data Fusion Pipeline
========================================
Ingests and aligns four heterogeneous sources onto a common spatio-temporal
grid so they can be stacked as channels into a single tensor for the model:

  1. Radar reflectivity  (DWR network, IMD)        -> 2D grid, ~10 min cadence
  2. Satellite IR/WV     (INSAT-3D/3DR, MOSDAC)      -> 2D grid, ~15 min cadence
  3. Lightning strikes   (CG/IC location network)    -> point events -> gridded density
  4. NWP model fields    (WRF/GFS: CAPE, PWAT, wind) -> 2D grid, 3 hr cadence

Real data source pointers (fill in credentials/endpoints for your team):
  - IMD radar: https://mausam.imd.gov.in (registered access) or IMD DWR HDF5 dumps
  - MOSDAC satellite: https://mosdac.gov.in (INSAT-3D/3DR L1/L2 products)
  - Lightning: IMD/ISRO lightning location network feed, or WWLLN as a proxy
  - NWP: NCMRWF/IMD WRF output, or open GFS via NOMADS as a substitute

This file gives you a working, runnable pipeline against SYNTHETIC data so the
team can develop the model end-to-end before real data access is sorted out —
swap `load_*` functions for real readers without touching anything downstream.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from normalization import normalize_channels

GRID_H, GRID_W = 128, 128          # common analysis grid (e.g. ~500x500 km at 4 km res)
TIME_STEP_MIN = 10                  # base cadence everything is resampled to
INPUT_STEPS = 6                     # 60 min of history as model input
OUTPUT_STEPS = 18                   # nowcast horizon: 18 x 10 min = 180 min
SIMULATION_SEED = 26072


@dataclass
class FrameStack:
    """One time step of fused, grid-aligned data. Shape: (H, W, C)."""
    radar_dbz: np.ndarray          # reflectivity (dBZ), missing = -32
    ir_temp: np.ndarray            # brightness temp (K), cold cloud tops = deep convection
    lightning_density: np.ndarray  # strikes per grid cell in the last TIME_STEP_MIN
    cape: np.ndarray               # convective available potential energy (J/kg)
    pwat: np.ndarray               # precipitable water (mm)

    def stack(self) -> np.ndarray:
        return np.stack(
            [self.radar_dbz, self.ir_temp, self.lightning_density, self.cape, self.pwat],
            axis=-1,
        ).astype(np.float32)


@dataclass
class StormCell:
    y: float
    x: float
    velocity_y: float
    velocity_x: float
    radius: float
    peak_dbz: float
    age: int
    lifetime: int


# ---------------------------------------------------------------------------
# Stateful synthetic simulator and public loaders.
# ---------------------------------------------------------------------------

class SyntheticDataSimulator:
    """Seeded, stateful storm simulator; one generated frame per timestamp."""

    def __init__(self, seed: int = SIMULATION_SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.yy, self.xx = np.mgrid[0:GRID_H, 0:GRID_W]
        self.cells: list[StormCell] = []
        self.frames: dict[int, tuple[FrameStack, list[dict]]] = {}
        self.first_step: int | None = None
        self.last_step: int | None = None
        self.cape: np.ndarray | None = None
        self.pwat: np.ndarray | None = None
        self.previous_cloud: np.ndarray | None = None

    @staticmethod
    def step_for(timestamp) -> int:
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return int(timestamp.timestamp() // (TIME_STEP_MIN * 60))
        return int(timestamp) // TIME_STEP_MIN

    def frame_at(self, timestamp) -> tuple[FrameStack, list[dict]]:
        step = self.step_for(timestamp)
        if step in self.frames:
            return self.frames[step]
        if self.last_step is None or step < self.first_step:
            self._reset(step)
        while self.last_step is None or self.last_step < step:
            next_step = self.first_step if self.last_step is None else self.last_step + 1
            self.frames[next_step] = self._make_frame(next_step)
            self.last_step = next_step
        return self.frames[step]

    def _reset(self, first_step: int) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.cells = []
        self.frames = {}
        self.first_step = first_step
        self.last_step = None
        self.cape = None
        self.pwat = None
        self.previous_cloud = None

    def _new_nwp(self) -> None:
        coarse_cape = self.rng.normal(900, 350, (16, 16))
        coarse_pwat = self.rng.normal(38, 8, (16, 16))
        self.cape = np.clip(gaussian_filter(np.kron(coarse_cape, np.ones((8, 8))), 5), 0, 3500).astype(np.float32)
        self.pwat = np.clip(gaussian_filter(np.kron(coarse_pwat, np.ones((8, 8))), 5), 5, 75).astype(np.float32)

    def _spawn_cell(self) -> None:
        speed = float(self.rng.uniform(2.0, 4.0))
        direction = float(self.rng.uniform(0, 2 * np.pi))
        self.cells.append(StormCell(
            y=float(self.rng.uniform(8, GRID_H - 8)),
            x=float(self.rng.uniform(8, GRID_W - 8)),
            velocity_y=speed * float(np.sin(direction)),
            velocity_x=speed * float(np.cos(direction)),
            radius=float(self.rng.uniform(5, 12)),
            peak_dbz=float(self.rng.uniform(45, 65)),
            age=0,
            lifetime=int(self.rng.integers(30, 55)),
        ))

    def _make_frame(self, step: int) -> tuple[FrameStack, list[dict]]:
        if step % 18 == 0 or self.cape is None:
            self._new_nwp()
        if not self.cells:
            for _ in range(3):
                self._spawn_cell()

        cape_factor = float(np.clip(np.mean(self.cape) / 1800, 0.15, 1.0))
        if self.rng.random() < 0.05 + 0.12 * cape_factor:
            self._spawn_cell()

        radar = self.rng.normal(-17.0, 2.0, (GRID_H, GRID_W))
        active: list[StormCell] = []
        for cell in self.cells:
            cell.age += 1
            cell.y += cell.velocity_y
            cell.x += cell.velocity_x
            life = cell.age / cell.lifetime
            lifecycle = min(1.0, cell.age / 5, max(0.0, (1.0 - life) / 0.3))
            radius = cell.radius * (0.8 + 0.35 * np.sin(np.pi * min(life, 1)))
            blob = np.exp(-((self.yy - cell.y) ** 2 + (self.xx - cell.x) ** 2) / (2 * radius ** 2))
            radar += cell.peak_dbz * lifecycle * blob
            if cell.age < cell.lifetime:
                active.append(cell)
        self.cells = active
        radar = np.clip(radar, -30, 75).astype(np.float32)

        cloud = gaussian_filter(np.maximum(radar - 5, 0), sigma=2.0)
        if self.previous_cloud is not None:
            cloud = 0.65 * cloud + 0.35 * self.previous_cloud
        self.previous_cloud = cloud
        ir = np.clip(290 - 2.5 * cloud + self.rng.normal(0, 1.2, cloud.shape), 180, 320).astype(np.float32)

        lightning = np.zeros((GRID_H, GRID_W), dtype=np.int32)
        events: list[dict] = []
        for cell in self.cells:
            local_intensity = max(0.0, (cell.peak_dbz - 25) / 40)
            count = int(self.rng.poisson(0.15 + 5.0 * local_intensity ** 2))
            if count == 0:
                continue
            ys = np.rint(self.rng.normal(cell.y, cell.radius * 0.38, count)).astype(int)
            xs = np.rint(self.rng.normal(cell.x, cell.radius * 0.38, count)).astype(int)
            in_grid = (ys >= 0) & (ys < GRID_H) & (xs >= 0) & (xs < GRID_W)
            for y, x in zip(ys[in_grid], xs[in_grid]):
                lightning[y, x] += 1
                events.append({
                    "timestamp": step * TIME_STEP_MIN * 60,
                    "lat": 5.0 + (GRID_H - 1 - y) * 30.0 / GRID_H,
                    "lon": 65.0 + x * 35.0 / GRID_W,
                    "grid_y": int(y),
                    "grid_x": int(x),
                    "strike_type": "CG" if self.rng.random() < 0.35 else "IC",
                })

        frame = FrameStack(radar, ir, lightning, self.cape.copy(), self.pwat.copy())
        return frame, events


_DEFAULT_SIMULATOR = SyntheticDataSimulator()


def load_radar(timestamp) -> np.ndarray:
    """Return simulated (GRID_H, GRID_W) reflectivity in dBZ."""
    return _DEFAULT_SIMULATOR.frame_at(timestamp)[0].radar_dbz.copy()


def load_satellite_ir(timestamp) -> np.ndarray:
    """Return simulated (GRID_H, GRID_W) IR brightness temperature in Kelvin."""
    return _DEFAULT_SIMULATOR.frame_at(timestamp)[0].ir_temp.copy()


def load_lightning(timestamp) -> np.ndarray:
    """Return integer strike counts in the preceding TIME_STEP_MIN."""
    return _DEFAULT_SIMULATOR.frame_at(timestamp)[0].lightning_density.copy()


def load_nwp(timestamp) -> Tuple[np.ndarray, np.ndarray]:
    """Return CAPE and PWAT fields held constant between 3-hour cycles."""
    frame = _DEFAULT_SIMULATOR.frame_at(timestamp)[0]
    return frame.cape.copy(), frame.pwat.copy()


def load_lightning_events(timestamp) -> list[dict]:
    """Return the raw synthetic events corresponding to a gridded frame."""
    return [event.copy() for event in _DEFAULT_SIMULATOR.frame_at(timestamp)[1]]


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------

def build_frame(timestamp, nwp_loader: Callable = load_nwp) -> FrameStack:
    cape, pwat = nwp_loader(timestamp)
    return FrameStack(
        radar_dbz=load_radar(timestamp),
        ir_temp=load_satellite_ir(timestamp),
        lightning_density=load_lightning(timestamp),
        cape=cape,
        pwat=pwat,
    )


def build_training_sequence(
    start_timestamp, nwp_loader: Callable = load_nwp
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X            (INPUT_STEPS, H, W, C)  fused observation history
      y_reflect    (OUTPUT_STEPS, H, W)    future radar reflectivity targets
      y_lightning  (OUTPUT_STEPS, H, W)    future lightning-occurrence probability targets (0/1 binned)
    """
    timestamps = range(INPUT_STEPS + OUTPUT_STEPS)
    if isinstance(start_timestamp, datetime):
        timestamps = [
            start_timestamp + timedelta(minutes=i * TIME_STEP_MIN) for i in timestamps
        ]
    else:
        timestamps = [start_timestamp + i * TIME_STEP_MIN for i in timestamps]
    frames = [build_frame(timestamp, nwp_loader=nwp_loader) for timestamp in timestamps]
    stacks = [f.stack() for f in frames]
    X = np.stack(stacks[:INPUT_STEPS], axis=0)
    y_reflect = np.stack([f.radar_dbz for f in frames[INPUT_STEPS:]], axis=0)
    y_lightning = np.stack(
        [(f.lightning_density > 0).astype(np.float32) for f in frames[INPUT_STEPS:]], axis=0
    )
    return X, y_reflect, y_lightning


def normalize(X: np.ndarray) -> np.ndarray:
    """Normalize channel-last inputs with statistics from the starter archive."""
    return normalize_channels(X)


if __name__ == "__main__":
    X, y_dbz, y_lightning = build_training_sequence(start_timestamp=0)
    print("Input tensor  X:", X.shape, " (steps, H, W, channels)")
    print("Target dBZ    :", y_dbz.shape)
    print("Target lightning:", y_lightning.shape)
    print("Channels: [radar_dbz, ir_temp, lightning_density, cape, pwat]")
