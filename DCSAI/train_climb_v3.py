from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from takeoff_env import TakeoffEnv


MODEL_DIR = Path("models")
LOG_DIR = Path("logs/takeoff_ppo_v3_climb")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Change this only if your verified 20/20 takeoff model has another name.
START_MODEL = MODEL_DIR / "takeoff_ppo_v2_final"

FINAL_MODEL = MODEL_DIR / "takeoff_ppo_v3_climb_final"
TOTAL_TIMESTEPS = 200_000

env = TakeoffEnv()

model = PPO.load(START_MODEL, env=env, device="cpu")

logger = configure(
    folder=str(LOG_DIR),
    format_strings=["stdout", "csv", "tensorboard"],
)
model.set_logger(logger)

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=str(MODEL_DIR),
    name_prefix="takeoff_ppo_v3_climb_checkpoint",
)

try:
    print(f"Continuing from: {START_MODEL}.zip")
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps.")
    print(f"TensorBoard logs: {LOG_DIR}")
    print("Do not close this window until training saves the final model.")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    model.save(FINAL_MODEL)
    print(f"Final model saved: {FINAL_MODEL}.zip")

finally:
    env.close()
