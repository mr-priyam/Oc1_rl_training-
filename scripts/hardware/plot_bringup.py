"""plot_bringup.py - simple plots of one bring-up CSV (target vs measured, torque vs rated).

    .venv/bin/python scripts/hardware/plot_bringup.py runs/hardware/<date>/policy_air_step.csv
    .venv/bin/python scripts/hardware/plot_bringup.py runs/hardware/<date>          every CSV in it

Saves <csv name>.png next to the CSV.  Each joint gets one panel with two graphs:
  top:    target (dashed) and measured (solid) angle - far apart = motor not following
  bottom: torque as % of rated, with the rated line at 100%
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="All-NaN")

JOINTS = ("right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee_pitch", "right_ankle_pitch",
          "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee_pitch", "left_ankle_pitch")
RATED = {"hip_pitch": 40, "hip_roll": 40, "knee_pitch": 40, "hip_yaw": 20, "ankle_pitch": 20}


def rated(j):
  return next(v for k, v in RATED.items() if j.endswith(k))


def plot(csv_path):
  d = np.genfromtxt(csv_path, delimiter=",", names=True)
  if d.size < 2:
    print(f"skip {csv_path} (empty)")
    return
  t = d["t"]
  fig, axes = plt.subplots(4, 5, figsize=(16, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1, 2, 1]))
  for i, j in enumerate(JOINTS):
    row, col = 2 * (i // 5), i % 5
    ax, axt = axes[row, col], axes[row + 1, col]
    ax.plot(t, d[f"target_{j}"], "--", color="tab:blue", lw=1.2, label="target")
    ax.plot(t, d[f"q_{j}"], color="black", lw=1.2, label="measured")
    err = np.nanmax(np.abs(d[f"q_{j}"] - d[f"target_{j}"]))
    ax.set_title(f"{j.replace('_', ' ')}   max error {err:.3f} rad", fontsize=9)
    ax.grid(alpha=0.3)
    pct = 100 * np.abs(d[f"tau_{j}"]) / rated(j)
    axt.plot(t, pct, color="tab:orange", lw=1)
    axt.axhline(100, color="red", lw=1)
    axt.set_ylim(0, max(120, np.nanmax(pct) * 1.1 if np.isfinite(pct).any() else 120))
    axt.grid(alpha=0.3)
    if col == 0:
      ax.set_ylabel("angle (rad)")
      axt.set_ylabel("torque %\nof rated")
  axes[0, 0].legend(fontsize=8)
  for ax in axes[-1]:
    ax.set_xlabel("time (s)")
  title = Path(csv_path).stem.replace("_", " ")
  if "tilt" in d.dtype.names and np.nanmax(d["tilt"]) > 0:
    title += f"    max body tilt {np.nanmax(d['tilt']):.1f} deg"
  fig.suptitle(title)
  fig.tight_layout()
  out = Path(csv_path).with_suffix(".png")
  fig.savefig(out, dpi=110)
  plt.close(fig)
  print(f"saved {out}")


def main():
  if len(sys.argv) < 2:
    sys.exit(__doc__)
  for arg in sys.argv[1:]:
    p = Path(arg)
    for f in (sorted(p.glob("*.csv")) if p.is_dir() else [p]):
      plot(f)


if __name__ == "__main__":
  main()
