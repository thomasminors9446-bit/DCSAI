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
TOTAL_TIMESTEPS = 1_000_000

# ----------------------------------------------------------------------
# Prepare output directories
# ----------------------------------------------------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Build the environment (head-less - no GUI needed)
# ----------------------------------------------------------------------
# Pass the waypoint at construction time; the Monitor wrapper records episode stats.
env = Monitor(TakeoffEnv(waypoint=WAYPOINT, log_dir=LOG_DIR, plot=False))

# ----------------------------------------------------------------------
# Check-point callback - saves a snapshot every 25 k steps
# ----------------------------------------------------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=str(MODEL_DIR),
    name_prefix="waypoint_ppo_checkpoint",
)

# ----------------------------------------------------------------------
# PPO hyper-parameters - same as the original take-off training script
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
    device="cpu",
    verbose=1,
    tensorboard_log=str(LOG_DIR),
    policy_kwargs={"net_arch": [256, 256]},
)

print(f"=== Starting PPO training for {TOTAL_TIMESTEPS:,} timesteps ===")
print(f"   Waypoint : {WAYPOINT}")
print(f"   Checkpoints saved in: {MODEL_DIR}")
print(f"   TensorBoard logs at: {LOG_DIR}\n")
print("Press Ctrl-C to stop early - the latest checkpoint will be saved.")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )
finally:
    model.save(MODEL_DIR / "waypoint_ppo_final")
    # ----------------------------------------------------------------------
    # Run a quick evaluation using the trained model (pure agent steering).
    # This evaluates without re‑loading the model from disk, avoiding zip‑read issues.
    # ----------------------------------------------------------------------
    eval_env = TakeoffEnv(
        waypoint=WAYPOINT,
        log_dir=Path(r"C:/Users/Thoma/DCSAI_repo/logs/eval_waypoint"),
        plot=False,
        use_autonomous_steering=False,
    )
    print("\n=== Evaluation of trained policy (5 episodes) ===")
    for ep in range(1, 6):
        obs, info = eval_env.reset(seed=ep)
        total_reward = 0.0
        for step in range(eval_env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            if terminated or truncated:
                success = info.get("success", False)
                final_dist = info.get("distance_to_wp", float("nan"))
                print(
                    f"Episode {ep:02d} finished after {step+1} steps; "
                    f"total_reward={total_reward:.2f}; success={success}; "
                    f"final_dist_to_wp={final_dist:.1f}m"
                )
                break
    eval_env.close()
    env.close()

print("\n=== Training finished ===")
print(f"Final model saved as: {MODEL_DIR / 'waypoint_ppo_final.zip'}")
