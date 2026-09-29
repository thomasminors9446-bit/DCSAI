# train_waypoint.py
# -------------------------------------------------
# Train a PPO agent to reach a fixed waypoint using the
# extended TakeoffEnv (which now includes waypoint distance
# and bearing observations, reward shaping, CSV logging and
# optional trajectory plotting).
# -------------------------------------------------

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from takeoff_env import TakeoffEnv

# ----------------------------------------------------------------------
# USER SETTINGS – edit these to experiment
# ----------------------------------------------------------------------
# Fixed waypoint (meters). Change as desired; the environment will
# reward the agent for reducing the distance to this point.
WAYPOINT = (2000.0, 0.0)  # 2 km straight ahead, no lateral offset

# Where to store model checkpoints and TensorBoard logs
MODEL_DIR = Path("models/waypoint")
LOG_DIR = Path("logs/waypoint_ppo")

# Total training timesteps – the default 300_000 is reasonable for a quick
# demo, but feel free to increase for better performance.
TOTAL_TIMESTEPS = 300_000

# ----------------------------------------------------------------------
# Prepare output directories
# ----------------------------------------------------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Build the environment (head‑less – no GUI needed)
# ----------------------------------------------------------------------
# Pass the waypoint at construction time; the Monitor wrapper records episode stats.
env = Monitor(TakeoffEnv(waypoint=WAYPOINT))

# ----------------------------------------------------------------------
# Check‑point callback – saves a snapshot every 25 k steps
# ----------------------------------------------------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=str(MODEL_DIR),
    name_prefix="waypoint_ppo_checkpoint",
)

# ----------------------------------------------------------------------
# PPO hyper‑parameters – same as the original take‑off training script
# ----------------------------------------------------------------------
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
    policy_kwargs={"net_arch": [128, 128]},
)

print(f"=== Starting PPO training for {TOTAL_TIMESTEPS:,} timesteps ===")
print(f"   Waypoint : {WAYPOINT}")
print(f"   Checkpoints saved in: {MODEL_DIR}")
print(f"   TensorBoard logs at: {LOG_DIR}\n")
print("Press Ctrl‑C to stop early – the latest checkpoint will be saved.")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )
finally:
    model.save(MODEL_DIR / "waypoint_ppo_final")
    env.close()

print("\n=== Training finished ===")
print(f"Final model saved as: {MODEL_DIR / 'waypoint_ppo_final.zip'}")
