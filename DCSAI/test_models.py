from pathlib import Path

from stable_baselines3 import PPO

models_dir = Path("models")
files = sorted(models_dir.glob("*.zip"))

if not files:
    raise SystemExit("No .zip model files found in F:\\DCSAI\\models")

for path in files:
    try:
        PPO.load(path, device="cpu")
        print(f"OK      {path.name}  ({path.stat().st_size:,} bytes)")
    except Exception as error:
        print(f"BROKEN  {path.name}  ({path.stat().st_size:,} bytes)")
        print(f"        {type(error).__name__}: {error}")