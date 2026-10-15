"""SQLite persistence and PyTorch access for VAJRA gridded sequences."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from torch.utils.data import Dataset
from normalization import normalize_channels

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_DB_URL = f"sqlite:///{(DEFAULT_DATA_DIR / 'vajra.sqlite3').as_posix()}"


class Base(DeclarativeBase):
    pass


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    source: Mapped[str] = mapped_column(String(24), index=True)
    grid_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class LightningEvent(Base):
    __tablename__ = "lightning_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    grid_y: Mapped[int] = mapped_column(Integer)
    grid_x: Mapped[int] = mapped_column(Integer)
    strike_type: Mapped[str] = mapped_column(String(2))


class SequenceRecord(Base):
    __tablename__ = "sequences"

    sequence_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(8))
    split: Mapped[str] = mapped_column(String(8), index=True)


_engine = None
_session_factory = None
_data_dir = DEFAULT_DATA_DIR
_pending_session = None


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _get_session():
    if _session_factory is None:
        init_db()
    return _session_factory()


def init_db(db_url: str | None = None, data_dir: str | Path | None = None):
    """Create the schema and configure this module's database and grid root."""
    global _engine, _session_factory, _data_dir, _pending_session
    if _pending_session is not None:
        _pending_session.close()
        _pending_session = None
    if _engine is not None:
        _engine.dispose()
    db_url = db_url or os.environ.get("VAJRA_DB_URL", DEFAULT_DB_URL)
    _data_dir = Path(data_dir or os.environ.get("VAJRA_DATA_DIR", DEFAULT_DATA_DIR)).resolve()
    _data_dir.mkdir(parents=True, exist_ok=True)
    if db_url.startswith("sqlite:///"):
        Path(db_url.removeprefix("sqlite:///" )).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(db_url)
    _session_factory = sessionmaker(_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    return _engine


def commit_database() -> None:
    """Commit any records added with ``commit=False``."""
    global _pending_session
    if _pending_session is not None:
        _pending_session.commit()
        _pending_session.close()
        _pending_session = None


def _writer():
    global _pending_session
    if _pending_session is None:
        _pending_session = _get_session()
    return _pending_session


def insert_frame(
    timestamp: datetime,
    source: str,
    grid: np.ndarray,
    grid_path: str | Path | None = None,
    *,
    commit: bool = True,
) -> Frame:
    """Save a grid as ``.npy`` and insert its frame metadata."""
    timestamp = _utc_naive(timestamp)
    if grid_path is None:
        safe_time = timestamp.strftime("%Y%m%dT%H%M%S")
        path = _data_dir / "grids" / source / f"{safe_time}.npy"
    else:
        path = Path(grid_path)
        if not path.is_absolute():
            path = PROJECT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(grid))
    try:
        relative_path = path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        relative_path = str(path.resolve())
    record = Frame(timestamp=timestamp, source=source, grid_path=relative_path)
    session = _writer()
    session.add(record)
    session.flush()
    if commit:
        session.commit()
        session.refresh(record)
        session.close()
        global _pending_session
        _pending_session = None
    return record


def _event_timestamp(value) -> datetime:
    if isinstance(value, datetime):
        return _utc_naive(value)
    return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)


def insert_lightning_events(events: Iterable[dict], *, commit: bool = True) -> int:
    """Insert event dictionaries returned by ``load_lightning_events``."""
    rows = [
        LightningEvent(
            timestamp=_event_timestamp(event["timestamp"]),
            lat=float(event["lat"]),
            lon=float(event["lon"]),
            grid_y=int(event["grid_y"]),
            grid_x=int(event["grid_x"]),
            strike_type=str(event["strike_type"]),
        )
        for event in events
    ]
    if rows:
        session = _writer()
        session.add_all(rows)
        if commit:
            session.commit()
            session.close()
            global _pending_session
            _pending_session = None
    return len(rows)


def insert_sequence(
    sequence_id: int,
    frame_id: int,
    role: str,
    split: str,
    *,
    commit: bool = True,
) -> SequenceRecord:
    """Associate one source frame with an input/target sequence role."""
    if role not in {"input", "target"}:
        raise ValueError("role must be 'input' or 'target'")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'")
    record = SequenceRecord(sequence_id=sequence_id, frame_id=frame_id, role=role, split=split)
    session = _writer()
    session.add(record)
    if commit:
        session.commit()
        session.close()
        global _pending_session
        _pending_session = None
    return record


def get_sequence(sequence_id: int) -> list[dict]:
    """Return sequence rows with frame metadata ordered by timestamp/source."""
    session = _get_session()
    rows = session.execute(
        select(SequenceRecord, Frame)
        .join(Frame, SequenceRecord.frame_id == Frame.id)
        .where(SequenceRecord.sequence_id == sequence_id)
        .order_by(Frame.timestamp, Frame.source)
    ).all()
    result = [
        {
            "sequence_id": seq.sequence_id,
            "frame_id": frame.id,
            "timestamp": frame.timestamp.replace(tzinfo=timezone.utc),
            "source": frame.source,
            "grid_path": frame.grid_path,
            "role": seq.role,
            "split": seq.split,
        }
        for seq, frame in rows
    ]
    session.close()
    return result


def query_by_timerange(
    start: datetime, end: datetime, source: str | None = None
) -> list[Frame]:
    """Return frames in the half-open UTC interval ``[start, end)``."""
    statement = select(Frame).where(
        Frame.timestamp >= _utc_naive(start), Frame.timestamp < _utc_naive(end)
    )
    if source is not None:
        statement = statement.where(Frame.source == source)
    statement = statement.order_by(Frame.timestamp, Frame.source)
    session = _get_session()
    result = list(session.scalars(statement))
    session.close()
    return result


class DBSequenceDataset(Dataset):
    """Read fused inputs and radar/lightning targets from persisted sequences."""

    def __init__(self, split: str = "train"):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        session = _get_session()
        self.sequence_ids = list(session.scalars(
            select(SequenceRecord.sequence_id)
            .where(SequenceRecord.split == split)
            .distinct()
            .order_by(SequenceRecord.sequence_id)
        ))
        session.close()

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, index: int):
        rows = get_sequence(self.sequence_ids[index])
        by_time: dict[datetime, dict[str, tuple[str, str]]] = {}
        for row in rows:
            by_time.setdefault(row["timestamp"], {})[row["source"]] = (row["grid_path"], row["role"])
        input_times = sorted(t for t, sources in by_time.items() if next(iter(sources.values()))[1] == "input")
        target_times = sorted(t for t, sources in by_time.items() if next(iter(sources.values()))[1] == "target")

        def load(timestamp: datetime, source: str) -> np.ndarray:
            path = Path(by_time[timestamp][source][0])
            if not path.is_absolute():
                path = PROJECT_DIR / path
            return np.load(path, allow_pickle=False)

        inputs = []
        for timestamp in input_times:
            radar = load(timestamp, "radar")
            ir = load(timestamp, "satellite")
            lightning = load(timestamp, "lightning")
            nwp = load(timestamp, "nwp")
            inputs.append(np.stack([radar, ir, lightning, nwp[0], nwp[1]], axis=-1))
        targets_radar = [load(timestamp, "radar") for timestamp in target_times]
        targets_lightning = [(load(timestamp, "lightning") > 0).astype(np.float32) for timestamp in target_times]
        x = normalize_channels(np.stack(inputs).astype(np.float32))
        x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2).float()
        return (
            x_tensor,
            torch.from_numpy(np.stack(targets_radar).astype(np.float32)),
            torch.from_numpy(np.stack(targets_lightning).astype(np.float32)),
        )
