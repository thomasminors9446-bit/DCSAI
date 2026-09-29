from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from takeoff_env import TakeoffEnv


MODEL_DIR = Path("models")
LOG_DIR = Path("logs/takeoff_ppo")

TOTAL_TIMESTEPS = 300_000

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

env = Monitor(TakeoffEnv())

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=str(MODEL_DIR),
    name_prefix="takeoff_ppo_v2_checkpoint",
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=str(LOG_DIR),
    policy_kwargs={
        "net_arch": [128, 128],
    },
)

print(f"Starting PPO training for {TOTAL_TIMESTEPS:,} timesteps...")
print("Checkpoints will be saved in: models")
print("Press Ctrl+C to stop early; saved checkpoints remain available.")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )
finally:
    model.save(MODEL_DIR / "takeoff_ppo_final")
    env.close()

print("Training finished.")
print("Final model saved as: models/takeoff_ppo_v2_final.zip")