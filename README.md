# DCSAI

This repository provides a reinforcement‑learning environment for aircraft take‑off and climb dynamics using OpenAI Gymnasium.

## New Waypoint Navigation Feature

The `TakeoffEnv` class now supports optional waypoint navigation. You can specify a target waypoint that the aircraft should fly toward after take‑off. The environment will:

* expose **distance‑to‑waypoint** and **bearing‑to‑waypoint** as additional observation dimensions,
* shape the reward to encourage progress toward the waypoint,
* consider waypoint proximity as part of the success criteria, and
* optionally log per‑step data to CSV and generate a Matplotlib trajectory plot.

### Running a Single Episode with a Waypoint

```bash
python DCSAI/test_takeoff_env.py \
    --config waypoint.json \
    --log-dir logs \
    --plot \
    --seed 42
```

* `--config` points to a JSON file containing the waypoint coordinates, e.g.:
```json
{ "x": 2000, "y": 0 }
```
* `--log-dir` enables CSV logging (and the plot if `--plot` is also supplied).
* `--plot` generates a PNG plot of the ground‑track and altitude profile after the episode finishes.
* If you omit `--config`, the environment behaves exactly as the original take‑off simulation.

### Backward Compatibility

All existing scripts (`train_climb_v3.py`, `evaluate_takeoff.py`, etc.) continue to work unchanged because the new constructor arguments have defaults. If no waypoint is supplied, the environment's observation space and dynamics are identical to the previous version.

---

*For more details on the environment's API and the new waypoint capabilities, refer to the docstring in `takeoff_env.py`.