from stable_baselines3.common.env_checker import check_env

from takeoff_env import TakeoffEnv

env = TakeoffEnv()
check_env(env, warn=True)
env.close()

print("Environment check completed.")
