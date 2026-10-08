#!/usr/bin/env python3
"""Post-training P1 audit using the unchanged official inference functions.

Run only after the PPO process has exited. Never launches training. All conditions
reuse the same reset seeds; torch noise is reseeded per episode for paired controls.
The position override is intentionally the official qpos-only implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime, timezone

OFFICIAL_COMMIT = "70ea0fe0e039846126bd0f87f38738b710a0d515"
SOURCE_SHA256 = {
    "train.py": "d131ea352507403e2eb33cd0d630ee597b899fca6686d9b0e12ee4b5bae78205",
    "inference.py": "ae88385e6d9a40ce9ad8d291f1ad7fa04738bcc6efd3ced686223e7b74d978e0",
    "config.py": "2ce4c1b5396792d4cd2ca57ef2904e0f177954172bb7f28a338b161f630384a7",
    "device_utils.py": "90f401c3811df00a1c7178d894e34f31fd43328ab75b48bb546d7f02636887de",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0, middle - half), min(1, middle + half)]


def summarize(rows):
    n = len(rows)
    successes = sum(r["success"] for r in rows)
    rewards = [r["reward"] for r in rows]
    return {"episodes": n, "successes": successes, "success_rate": successes / n,
            "success_rate_wilson_95": wilson(successes, n),
            "grasps": sum(r["grasped"] for r in rows),
            "grasp_rate": sum(r["grasped"] for r in rows) / n,
            "grasp_rate_wilson_95": wilson(sum(r["grasped"] for r in rows), n),
            "mean_reward": statistics.mean(rewards),
            "reward_sample_std": statistics.stdev(rewards) if n > 1 else None,
            "mean_final_error_m": statistics.mean(r["final_pos_err_m"] for r in rows),
            "early_terminations": sum(r["terminated_early"] for r in rows)}


def compare(base, other):
    assert [r["reset_seed"] for r in base] == [r["reset_seed"] for r in other]
    bs, ts = summarize(base), summarize(other)
    return {"course_primary_metric": "grasp_rate",
            "grasp_rate_difference": ts["grasp_rate"] - bs["grasp_rate"],
            "grasp_rate_ratio": ts["grasp_rate"] / bs["grasp_rate"] if bs["grasp_rate"] else None,
            "target_grasps_for_half": bs["grasps"] / 2 if bs["grasps"] else None,
            "within_one_grasp_of_half": abs(ts["grasps"] - bs["grasps"] / 2) <= 1 if bs["grasps"] else None,
            "nonzero_approximately_half_grasps": 0 < ts["grasps"] < bs["grasps"] and abs(ts["grasps"] - bs["grasps"] / 2) <= 1 if bs["grasps"] else None,
            "baseline_grasp_to_no_grasp": sum(a["grasped"] and not b["grasped"] for a, b in zip(base, other)),
            "baseline_no_grasp_to_grasp": sum(not a["grasped"] and b["grasped"] for a, b in zip(base, other)),
            "success_rate_difference": ts["success_rate"] - bs["success_rate"],
            "mean_paired_reward_difference": statistics.mean(
                b["reward"] - a["reward"] for a, b in zip(base, other)),
            "baseline_success_to_failure": sum(a["success"] and not b["success"] for a, b in zip(base, other)),
            "baseline_failure_to_success": sum(not a["success"] and b["success"] for a, b in zip(base, other)),
            "success_rate_ratio": ts["success_rate"] / bs["success_rate"] if bs["success_rate"] else None,
            "empirical_half_sr_degradation": ts["success_rate"] <= bs["success_rate"] / 2 if bs["success_rate"] else None}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("./course_workspace/ap-physicalai-1"))
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("checkpoints", "conditions"), default="checkpoints")
    p.add_argument("--training-exit-file", type=Path, required=True,
                   help="Existing file containing 0, written only after verifying PPO process exit")
    p.add_argument("--training-log", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[50, 500, 1000])
    p.add_argument("--video-episodes", type=int, default=3)
    p.add_argument("--conditions", type=Path,
                   help="JSON array: name, cube_x and/or cube_y (absolute metres), obs_noise; baseline added")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    a.repo = a.repo.resolve()
    a.run_dir = a.run_dir.resolve() if a.run_dir else a.repo / "logs/my_first_run"
    a.out = a.out.resolve()
    if a.episodes < 1 or not 0 <= a.video_episodes <= a.episodes:
        p.error("Require episodes >= 1 and 0 <= video-episodes <= episodes")
    if a.phase == "conditions" and a.conditions is None:
        p.error("Conditions phase requires an explicit --conditions JSON file")
    return a


def read_conditions(path):
    rows = [{"name": "baseline", "cube_x": None, "cube_y": None, "obs_noise": 0.0}]
    if path is None:
        return rows
    names = {"baseline"}
    for raw in json.loads(path.read_text()):
        if set(raw) - {"name", "cube_x", "cube_y", "obs_noise"}:
            raise ValueError(f"Unknown condition keys: {raw}")
        c = {"cube_x": None, "cube_y": None, "obs_noise": 0.0, **raw}
        name = c["name"]
        if not name or not all(x.isalnum() or x in "_-" for x in name) or name in names:
            raise ValueError(f"Unsafe or duplicate condition name: {name}")
        for key in ("cube_x", "cube_y", "obs_noise"):
            if c[key] is not None and not math.isfinite(float(c[key])):
                raise ValueError(f"Nonfinite condition value: {key}")
        if c["obs_noise"] is None or c["obs_noise"] < 0:
            raise ValueError("Noise must be nonnegative")
        if c["obs_noise"] and (c["cube_x"] is not None or c["cube_y"] is not None):
            raise ValueError("Do not combine position and noise changes in one condition")
        names.add(name)
        rows.append(c)
    return rows


def verify_training(a, torch):
    assert a.training_exit_file.read_text().strip() == "0", "PPO has not been verified to exit successfully"
    verified_sources = {name: sha256(a.repo / name) for name in SOURCE_SHA256}
    assert verified_sources == SOURCE_SHA256, "Official source mismatch: audit before proceeding"
    checkpoints = {}
    for step in sorted(set(a.checkpoint_steps + [1000])):
        p = a.run_dir / f"model_{step}.pt"
        assert p.is_file() and p.stat().st_size > 0, f"Missing checkpoint {p}"
        c = torch.load(p, map_location="cpu", weights_only=False)
        assert c["iter"] == step, f"Checkpoint iteration mismatch: {p}"
        assert "actor_state_dict" in c and "critic_state_dict" in c
        assert all(bool(torch.isfinite(t).all()) for t in c["actor_state_dict"].values()), "Nonfinite actor weights"
        checkpoints[str(step)] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p),
                                  "saved_iteration": c["iter"], "keys": sorted(c)}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    series = []
    tags = {}
    for path in sorted(a.run_dir.glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        tags[path.name] = acc.Tags().get("scalars", [])
        if "Train/mean_reward" in tags[path.name]:
            series.extend({"step": e.step, "value": e.value, "wall_time": e.wall_time}
                          for e in acc.Scalars("Train/mean_reward"))
    series.sort(key=lambda r: (r["step"], r["wall_time"]))
    last = series[-1] if series else None
    return {"source_commit": OFFICIAL_COMMIT, "source_file_sha256": verified_sources,
            "training_process_exit": 0, "checkpoints": checkpoints,
            "config": json.loads((a.run_dir / "config.json").read_text()),
            "training_log_sha256": sha256(a.training_log) if a.training_log else None,
            "reward_tag": "Train/mean_reward", "available_scalar_tags": tags,
            "last_training_mean_reward": last,
            "max_training_mean_reward": max(series, key=lambda r: r["value"]) if series else None,
            "last_training_reward_at_least_1100": last["value"] >= 1100 if last else None,
            "reward_series": series,
            "warning": "Training mean reward and evaluation episode reward are distinct measurements."}


def save_video(env, trajectory, path):
    import imageio.v2 as iio
    import numpy as np
    fps = 1.0 / env.dt / 2
    frames_count = 0
    # Bound host RAM: render at most ten selected states per batch, stream MP4.
    selected = trajectory[::2]
    with iio.get_writer(path, fps=fps, codec="libx264", macro_block_size=16) as writer:
        for start in range(0, len(selected), 10):
            for frame in env.render(selected[start:start + 10], height=480, width=640):
                writer.append_data(np.asarray(frame))
                frames_count += 1
    with iio.get_reader(path) as reader:
        decoded = reader.count_frames()
        shape = list(reader.get_data(0).shape)
    assert decoded == frames_count and decoded > 0 and shape == [480, 640, 3]
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "frames": frames_count, "fps": fps, "decoded_shape": shape,
            "verified_decodable": True, "manual_visual_review": "PENDING"}


def main():
    a = parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit("Output directory is not empty; use a new directory to avoid mixing attempts")
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(a.repo))
    import jax
    import numpy as np
    import torch
    from importlib.metadata import version
    import inference as official
    from mujoco_playground import registry
    from device_utils import apply_mjx_overrides_to_playground_cfg
    training = verify_training(a, torch)
    write_json(a.out / "training_verification.json", training)
    conditions = read_conditions(a.conditions) if a.phase == "conditions" else read_conditions(None)
    steps = a.checkpoint_steps if a.phase == "checkpoints" else [1000]
    cfg = registry.get_default_config(official.ENV_NAME)
    apply_mjx_overrides_to_playground_cfg(cfg)
    env = registry.load(official.ENV_NAME, config=cfg)
    jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
    is_dict_obs = isinstance(env.observation_size, dict)
    runner = official.build_runner(str(a.run_dir / f"model_{steps[0]}.pt"), a.device)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_commit": OFFICIAL_COMMIT,
        "phase": a.phase, "seeds": list(range(a.seed, a.seed + a.episodes)),
        "noise_rng": "torch.manual_seed(reset_seed) immediately before each episode",
        "policy": "official deterministic get_inference_policy; action clipping [-1,1] unchanged",
        "rollout": "unchanged inference.rollout_single_episode", "episode_length": int(cfg.episode_length),
        "control_dt_s": float(env.dt), "environment_config": cfg.to_dict(),
        "versions": {p: version(p) for p in ("torch", "jax", "jaxlib", "mujoco", "mujoco-mjx", "playground", "rsl-rl-lib")},
        "torch_deviation": "2.8.0/cu128 replaces course 2.6.0/cu124 for JAX cuDNN compatibility",
        "course_primary_metric": "grasp count out of 10 episodes (grasp_rate), not placement success rate",
        "lesson_verified": "https://practicum.yandex.ru/learn/ap-physicalai-st/courses/ae1aac0a-7c74-44bb-a346-074bea4e29ea/sprints/1085144/topics/f7692267-4be8-470d-8ca1-931bed77ef2f/lessons/cc836c35-48da-4b2f-a1e2-2cb643560afe/",
        "lesson_coordinate_mismatch": "Lesson describes standard cube near (0,0.5) and safe axes -0.3..0.3; pinned Playground actually uses home (0.7,0,0.03) with x offset +/-0.2. Literal lesson examples are retained as absolute coordinates and not silently transposed.",
        "success_definition": f"final cube-to-target Euclidean distance < {official.SUCCESS_THRESHOLD} m",
        "grasp_definition": f"max cube z > {official.GRASP_Z_THRESHOLD} m; this does not itself mean success",
        "position_semantics": "cube_x/cube_y are absolute metres, not offsets; unspecified axis keeps paired random reset",
        "position_override_caveat": "Official override changes qpos only: first observation and derived MJX fields are stale until first step. Preserved for course fidelity; do not call this a fully corrected physical initial-state intervention.",
        "init_object_position": np.asarray(env._init_obj_pos).tolist(),
        "training_box_x_range_m": [float(env._init_obj_pos[0]) - .2, float(env._init_obj_pos[0]) + .2],
        "training_box_y_range_m": [float(env._init_obj_pos[1]) - .2, float(env._init_obj_pos[1]) + .2],
        "out_of_bounds": "step ends episode if any |box coordinate| > 1 m or box z < 0; exact +/-1 are boundary probes",
        "conditions": conditions, "manual_visual_review": "PENDING; generated video is not evidence a person inspected it",
        "small_sample_caveat": "10 episodes gives a coarse empirical estimate; approximately half baseline grasps (+/-1 grasp) is the lesson target, not a proven robustness boundary",
    }
    write_json(a.out / "metadata.json", metadata)
    results = []
    summaries = []
    video_errors = []
    started = time.time()
    for step in steps:
        if step != steps[0]:
            runner.load(str(a.run_dir / f"model_{step}.pt"), map_location=a.device)
        policy = runner.get_inference_policy(device=a.device)
        baseline = None
        for cond in conditions:
            case_dir = a.out / f"model_{step}" / cond["name"]
            case_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for ep in range(a.episodes):
                seed = a.seed + ep
                torch.manual_seed(seed)
                trajectory, reward, grasped, success, err = official.rollout_single_episode(
                    env, policy, jax.random.PRNGKey(seed), cfg.episode_length, is_dict_obs,
                    jit_reset=jit_reset, jit_step=jit_step, cube_x=cond["cube_x"],
                    cube_y=cond["cube_y"], obs_noise=cond["obs_noise"])
                assert all(math.isfinite(v) for v in (reward, err)), "Nonfinite evaluation result"
                row = {"checkpoint_step": step, "condition": cond["name"], "episode": ep + 1,
                       "reset_seed": seed, "noise_seed": seed, "reward": float(reward),
                       "grasped": bool(grasped), "success": bool(success), "final_pos_err_m": float(err),
                       "steps": len(trajectory) - 1, "terminated_early": len(trajectory) - 1 < cfg.episode_length,
                       "initial_cube_xyz_m": official._box_pos(trajectory[0]).tolist(),
                       "final_cube_xyz_m": official._box_pos(trajectory[-1]).tolist(),
                       "target_xyz_m": official._target_pos(trajectory[0]).tolist(),
                       "max_cube_z_m": max(official._cube_z(s) for s in trajectory),
                       "final_out_of_bounds": float(trajectory[-1].metrics.get("out_of_bounds", 0)),
                       "cube_x": cond["cube_x"], "cube_y": cond["cube_y"], "obs_noise": cond["obs_noise"]}
                np.savez_compressed(case_dir / f"episode_{ep + 1:02d}_seed_{seed}.npz",
                    qpos=np.stack([np.asarray(s.data.qpos) for s in trajectory]),
                    qvel=np.stack([np.asarray(s.data.qvel) for s in trajectory]),
                    ctrl=np.stack([np.asarray(s.data.ctrl) for s in trajectory]),
                    target_pos=np.asarray(trajectory[0].info["target_pos"]),
                    reward=np.asarray([float(s.reward) for s in trajectory]), reset_seed=seed)
                if ep < a.video_episodes:
                    path = case_dir / f"episode_{ep + 1:02d}_seed_{seed}.mp4"
                    try:
                        row["video"] = save_video(env, trajectory, path)
                    except Exception as error:
                        row["video_error"] = repr(error)
                        video_errors.append({"path": str(path), "error": repr(error)})
                        print("VIDEO_FAILED", str(path), repr(error), flush=True)
                rows.append(row)
                results.append(row)
                with (a.out / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                print("EPISODE", json.dumps({k: row[k] for k in ("checkpoint_step", "condition", "episode", "reset_seed", "reward", "success", "grasped", "final_pos_err_m")}), flush=True)
                del trajectory
            summary = {"checkpoint_step": step, **cond, **summarize(rows)}
            if cond["name"] == "baseline":
                baseline = rows
            else:
                summary["paired_comparison_to_baseline"] = compare(baseline, rows)
            video_paths = [Path(r["video"]["path"]) for r in rows if "video" in r]
            if video_paths and len(video_paths) == a.video_episodes:
                import subprocess
                import imageio_ffmpeg
                import imageio.v2 as iio
                # All per-episode streams have identical frame rate/size/codec.
                concat_list = case_dir / "video_concat.txt"
                concat_list.write_text("".join("file '" + p.name + "'\n" for p in video_paths))
                name = ("exp1_indomain" if cond["name"] == "baseline" else cond["name"]) if a.phase == "conditions" else f"model_{step}"
                combined = a.out / (name + ".mp4")
                try:
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(combined)], check=True)
                    with iio.get_reader(combined) as reader:
                        decoded = reader.count_frames()
                    expected = sum(r["video"]["frames"] for r in rows if "video" in r)
                    assert decoded == expected and decoded > 0
                    summary["combined_video"] = {"path": str(combined), "episodes": len(video_paths),
                        "frames": decoded, "sha256": sha256(combined), "verified_decodable": True,
                        "manual_visual_review": "PENDING"}
                except Exception as error:
                    video_errors.append({"path": str(combined), "error": repr(error)})
            summaries.append(summary)
            write_json(a.out / "summary.json", {"status": "RUNNING", "conditions": summaries})
            print("CONDITION", json.dumps(summary), flush=True)
            if a.phase == "conditions" and cond["name"] == "baseline" and summary["grasps"] == 0:
                write_json(a.out / "summary.json", {"status": "BLOCKED_BASELINE_NO_GRASPS", "conditions": summaries,
                    "reason": "Zero baseline grasps makes a half-grasp degradation target undefined; no OOD trials run."})
                return 3
    write_json(a.out / "summary.json", {"status": "METRICS_COMPLETE" if video_errors else "METRICS_AND_VIDEO_FILES_COMPLETE",
        "manual_visual_review": "PENDING", "conditions": summaries, "video_errors": video_errors,
        "elapsed_seconds": time.time() - started,
        "training_last_reward_at_least_1100": training["last_training_reward_at_least_1100"]})
    csv_fields = [k for k, v in results[0].items() if not isinstance(v, (list, dict)) and k != "video_error"]
    with (a.out / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    write_json(a.out / "artifact_manifest.json", [{"path": str(p.relative_to(a.out)), "bytes": p.stat().st_size,
               "sha256": sha256(p)} for p in sorted(a.out.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"])
    print("P1_EVALUATION_COMPLETE", a.out, "manual_visual_review=PENDING", flush=True)
    return 2 if video_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
