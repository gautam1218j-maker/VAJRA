"""
Sanity-check plot: pulls one sequence from the SQLite archive and renders the
radar reflectivity channel across every input + output step, so you can SEE
a storm drift/grow/decay across frames rather than just trusting a
correlation number.

Usage:
    python plot_sequence.py                      # first train sequence
    python plot_sequence.py --split val --index 3
    python plot_sequence.py --lightning          # overlay lightning strikes too

Requires: pip install matplotlib
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import db
from normalization import load_normalization_stats

RADAR_CHANNEL = 0  # channel order: [radar, ir, lightning, cape, pwat]
LIGHTNING_CHANNEL = 2

DB_PATH = Path("data/starter_dataset").resolve()

# Forecast targets are unnormalized, so input channels are restored with the
# same archive statistics used by DBSequenceDataset.
CHANNEL_MEAN, CHANNEL_STD = load_normalization_stats()


def load_one_sequence(split: str, index: int):
    db.init_db(f"sqlite:///{(DB_PATH / 'vajra.sqlite3').as_posix()}", DB_PATH)
    loader = DataLoader(db.DBSequenceDataset(split=split), batch_size=1, shuffle=False)
    for i, (X, y_reflect, y_lightning) in enumerate(loader):
        if i == index:
            # X: (1, T_in, C, H, W) -> (T_in, H, W); y_*: (1, T_out, H, W) -> (T_out, H, W)
            return X[0].numpy(), y_reflect[0].numpy(), y_lightning[0].numpy()
    raise IndexError(f"split {split!r} has fewer than {index + 1} sequences")


def plot_sequence(X, y_reflect, y_lightning, show_lightning: bool, out_path: Path):
    t_in = X.shape[0]
    t_out = y_reflect.shape[0]
    total = t_in + t_out
    cols = 6
    rows = -(-total // cols)  # ceil

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    vmin, vmax = -30, 75  # physically plausible dBZ range, fixed so frames are comparable

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])

        if idx >= total:
            ax.axis("off")
            continue

        if idx < t_in:
            # X comes out of DBSequenceDataset already normalized — undo that
            # per-channel so it's on the same raw physical scale as the targets.
            radar = X[idx, RADAR_CHANNEL] * CHANNEL_STD[RADAR_CHANNEL] + CHANNEL_MEAN[RADAR_CHANNEL]
            if show_lightning:
                lightning = X[idx, LIGHTNING_CHANNEL] * CHANNEL_STD[LIGHTNING_CHANNEL] + CHANNEL_MEAN[LIGHTNING_CHANNEL]
            else:
                lightning = None
            label = f"in t-{(t_in - idx) * 10}m"
        else:
            step = idx - t_in
            radar = y_reflect[step]
            lightning = y_lightning[step] if show_lightning else None
            label = f"fcst t+{(step + 1) * 10}m"

        ax.imshow(radar, cmap="turbo", vmin=vmin, vmax=vmax)
        if show_lightning and lightning is not None and np.any(lightning > 0):
            ys, xs = np.nonzero(lightning > (0 if lightning.max() <= 1 else 0.5))
            ax.scatter(xs, ys, s=4, c="white", marker="x", linewidths=0.6)
        ax.set_title(label, fontsize=8, color="white" if idx < t_in else "yellow")

    fig.suptitle("Radar reflectivity (dBZ) — history (white titles) then forecast (yellow titles)",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="black")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0, help="which sequence in the split to plot")
    parser.add_argument("--lightning", action="store_true", help="overlay lightning strike markers")
    parser.add_argument("--out", default="sequence_check.png")
    args = parser.parse_args()

    X, y_reflect, y_lightning = load_one_sequence(args.split, args.index)
    plot_sequence(X, y_reflect, y_lightning, args.lightning, Path(args.out))
