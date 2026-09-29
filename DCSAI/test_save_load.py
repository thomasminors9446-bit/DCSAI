from pathlib import Path

from stable_baselines3 import PPO

from takeoff_env import TakeoffEnv


output = Path("models") / "save_load_smoke_test"
output.parent.mkdir(parents=True, exist_ok=True)

env = TakeoffEnv()

try:
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=64,
        verbose=1,
        device="cpu",
    )

    model.learn(total_timesteps=128)

    model.save(output)
    print(f"Saved: {output}.zip")

    restored = PPO.load(output, env=env, device="cpu")
    print("SUCCESS: Model saved and loaded correctly.")

finally:
    env.close()