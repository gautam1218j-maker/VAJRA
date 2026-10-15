# VAJRA Synthetic Data Schema

## Starter Archive

The checked-in generator creates a deterministic four-day archive with a fixed UTC start of `2024-06-01T00:00:00Z` and seed `26072`. It contains 576 timestamps at 10-minute cadence on a 128 x 128 grid. Recreate it from the project root with:

```powershell
.venv\Scripts\python.exe generate_starter_dataset.py
```

Output is written to `data/starter_dataset/`:

```text
data/starter_dataset/
  vajra.sqlite3
  manifest.json
  grids/
    radar/YYYYMMDDTHHMMSS.npy
    satellite/YYYYMMDDTHHMMSS.npy
    lightning/YYYYMMDDTHHMMSS.npy
    nwp/YYYYMMDDTHHMMSS.npy
```

Each grid path is stored relative to the project root. Radar and satellite arrays are `float32` with shape `(128, 128)`. Lightning arrays are non-negative integer strike counts with shape `(128, 128)`. NWP arrays are `float32` with shape `(2, 128, 128)`; index 0 is CAPE in J/kg and index 1 is PWAT in mm. Lightning events are also retained individually in the database. NWP arrays are held constant between 3-hour forecast cycles.

## Tables

### `frames`

One row per source grid and observation time. The archive has four rows per timestamp.

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | integer, primary key | Frame record identifier |
| `timestamp` | UTC datetime | Observation time, stored as timezone-naive UTC in SQLite |
| `source` | string | `radar`, `satellite`, `lightning`, or `nwp` |
| `grid_path` | string | `.npy` path relative to the project, or an absolute path for external data roots |
| `created_at` | UTC datetime | Row creation time |

### `lightning_events`

One row per synthetic CG or IC event; `grid_y` and `grid_x` identify the cell used to bin that event into its lightning grid.

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | integer, primary key | Event identifier |
| `timestamp` | UTC datetime | Event time |
| `lat`, `lon` | float | Synthetic event coordinates |
| `grid_y`, `grid_x` | integer | Grid cell coordinates |
| `strike_type` | string | `CG` or `IC` |

### `sequences`

Membership table for overlapping 24-step samples. Each sequence has six input timestamps and 18 target timestamps, with one membership row per source frame at every timestamp.

| Column | Type | Meaning |
| --- | --- | --- |
| `sequence_id` | integer | Identifies one training sample |
| `frame_id` | integer, foreign key | References `frames.id` |
| `role` | string | `input` for the first six steps, `target` for the following 18 |
| `split` | string | `train`, `val`, or `test` |

The composite primary key is (`sequence_id`, `frame_id`). Date splits are contiguous and half-open: train is June 1 through June 2, validation is June 3, and test is June 4 (all UTC). A sample is only included if its entire window fits inside its split. The generated archive has 265 train, 121 validation, and 121 test sequences.

## Loading A Batch

Initialize the archive database before constructing the dataset. The returned batch matches `train.py` conventions: fused inputs are normalized and channel-first, while radar and lightning targets retain their 18-step time axis.

```python
from pathlib import Path
from torch.utils.data import DataLoader
from db import DBSequenceDataset, init_db

archive = Path("data/starter_dataset").resolve()
init_db(f"sqlite:///{(archive / 'vajra.sqlite3').as_posix()}", data_dir=archive)
loader = DataLoader(DBSequenceDataset(split="train"), batch_size=4, shuffle=True)
X, y_reflect, y_lightning = next(iter(loader))
```

Tensor shapes are `X: (B, 6, 5, 128, 128)`, `y_reflect: (B, 18, 128, 128)`, and `y_lightning: (B, 18, 128, 128)`.
