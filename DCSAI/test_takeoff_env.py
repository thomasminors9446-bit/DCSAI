import argparse
import json
from pathlib import Path

from takeoff_env import TakeoffEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single TakeoffEnv episode with optional waypoint configuration.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON file containing waypoint coordinates {\"x\": float, \"y\": float}.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory where CSV logs (and optional trajectory plot) will be written.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a Matplotlib trajectory plot after the episode ends (requires --log-dir).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the environment reset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load waypoint from JSON config if supplied
    waypoint = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Waypoint config file not found: {config_path}")
        with config_path.open() as f:
            data = json.load(f)
            if "x" not in data or "y" not in data:
                raise ValueError("Waypoint JSON must contain 'x' and 'y' fields.")
            waypoint = (float(data["x"]), float(data["y"]))

    env = TakeoffEnv(
        waypoint=waypoint,
        log_dir=args.log_dir,
        plot=args.plot,
        use_autonomous_steering=False,  # user prefers pure agent steering
    )

    observation, _ = env.reset(seed=args.seed)
    print("Initial observation:", observation)

    for step in range(env.max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step:03d} reward={reward:+.3f} speed={info['speed_mps']:.1f}m/s "
            f"altitude={info['altitude_m']:.1f}m distance_wp={info.get('distance_to_wp',0):.1f}m"
        )
        if terminated or truncated:
            print("Episode ended:", info)
            break

    # Ensure logs are flushed and plot generated (if requested)
    env.close()


if __name__ == "__main__":
    main()
