from pathlib import Path

from stable_baselines3 import PPO

from takeoff_env import TakeoffEnv


MODEL_PATH = Path("models/takeoff_ppo_v3_climb_final")
EPISODES = 20

model = PPO.load(MODEL_PATH, device="cpu")

successes = 0
rewards = []
episode_lengths = []
final_altitudes = []
final_speeds = []
final_vertical_speeds = []

for episode in range(1, EPISODES + 1):
    env = TakeoffEnv()
    observation, _ = env.reset(seed=20_000 + episode)

    total_reward = 0.0
    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    success = bool(info.get("success", False))
    successes += int(success)

    rewards.append(total_reward)
    episode_lengths.append(env.steps)
    final_altitudes.append(info["altitude_m"])
    final_speeds.append(info["speed_mps"])
    final_vertical_speeds.append(info["vertical_speed_mps"])

    result = "SUCCESS" if success else "FAILED"
    print(
        f"Episode {episode:02d}: {result:<7} "
        f"steps={env.steps:4d} "
        f"reward={total_reward:9.2f} "
        f"altitude={info['altitude_m']:7.1f} m "
        f"climb={info['vertical_speed_mps']:6.2f} m/s "
        f"speed={info['speed_mps']:5.1f} m/s "
        f"lateral={info['lateral_m']:+6.2f} m"
    )

    env.close()

print()
print("========== CLIMB v3 SUMMARY ==========")
print(
    f"Success rate: {successes}/{EPISODES} "
    f"({100.0 * successes / EPISODES:.1f}%)"
)
print(f"Mean reward: {sum(rewards) / EPISODES:.2f}")
print(f"Mean steps: {sum(episode_lengths) / EPISODES:.1f}")
print(f"Mean final altitude: {sum(final_altitudes) / EPISODES:.2f} m")
print(f"Mean final speed: {sum(final_speeds) / EPISODES:.2f} m/s")
print(
    "Mean final climb rate: "
    f"{sum(final_vertical_speeds) / EPISODES:.2f} m/s"
)
