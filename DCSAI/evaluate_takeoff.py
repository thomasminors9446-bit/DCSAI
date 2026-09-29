from stable_baselines3 import PPO

from takeoff_env import TakeoffEnv


MODEL_PATH = "models/takeoff_ppo_final"

env = TakeoffEnv()
model = PPO.load(MODEL_PATH)

for episode in range(10):
    observation, info = env.reset(seed=episode)
    total_reward = 0.0

    print(f"\nEpisode {episode + 1}")

    for step in range(1000):
        action, _ = model.predict(
            observation,
            deterministic=True,
        )
        if step % 20 == 0:
            print(f"action={action}")
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0 or terminated or truncated:
            print(
                f"step={step:04d} "
                f"speed={info['speed_mps']:6.1f} m/s "
                f"altitude={info['altitude_m']:6.1f} m "
                f"lateral={info['lateral_m']:+6.1f} m "
                f"reward={total_reward:+8.2f}"
            )

        if terminated or truncated:
            outcome = "SUCCESS" if info["success"] else "FAILED"
            print(
                f"{outcome}: "
                f"steps={step + 1}, "
                f"total_reward={total_reward:.2f}, "
                f"final_altitude={info['altitude_m']:.1f} m"
            )
            break

env.close()