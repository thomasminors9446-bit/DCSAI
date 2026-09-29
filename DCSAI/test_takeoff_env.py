from takeoff_env import TakeoffEnv

env = TakeoffEnv()

observation, info = env.reset(seed=42)

print("Initial observation:", observation)

for step in range(20):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step:02d} "
        f"reward={reward:+.3f} "
        f"speed={info['speed_mps']:.1f} m/s "
        f"altitude={info['altitude_m']:.1f} m"
    )

    if terminated or truncated:
        print("Episode ended:", info)
        break