from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
PRETRAINED = ROOT / "pretrained" / "policy.onnx"


def latest_run() -> Path:
  runs = sorted(p for p in RUNS_DIR.glob("*") if (p / "policy.onnx").exists())
  if not runs:
    raise FileNotFoundError(f"No trained runs with policy.onnx in {RUNS_DIR}")
  return runs[-1]


def resolve_policy(arg: str) -> Path:
  if arg == "latest":
    try:
      return latest_run() / "policy.onnx"
    except FileNotFoundError:
      if PRETRAINED.exists():
        return PRETRAINED
      raise
  p = Path(arg)
  return p / "policy.onnx" if p.is_dir() else p