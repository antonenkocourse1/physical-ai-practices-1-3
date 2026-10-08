#!/usr/bin/env python3
"""Evaluate one explicit P2 checkpoint with unchanged official rollout semantics.

Evaluation always uses ten paired seeds (42--51), the common default reward,
and videos of episodes 1--3. It never trains, selects a checkpoint, alters course
sources, installs packages, or launches another process. Run under the launcher's
serial GPU lock after training exits. Only load trusted course checkpoints.

The training exit file is a supplied process-exit assertion, not standalone proof
of which process ran. An optional launcher's training record is checked against
the chosen files; absent proof is explicitly reported rather than manufactured.
Importing this module, --help, and the static tests use only the standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
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
SEEDS = tuple(range(42, 52))
VIDEO_EPISODES = 3
RENDER_EVERY = 2
RENDER_BATCH_STATES = 10
DEFAULT_REWARD_SCALES = {
    "gripper_box": 4.0, "box_target": 8.0,
    "no_floor_collision": 0.25, "robot_target_qpos": 0.3,
}
EVALUATION_REWARD_LABEL = "common_default_evaluation_episode_return"
TRAINING_REWARD_LABEL = "training_mean_reward_under_training_reward_config"


def require(condition, message):
    """An audit guard that remains active under python -O."""
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_metadata(path):
    path = Path(path).resolve()
    require(path.is_file() and path.stat().st_size > 0, f"Missing or empty file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Explicit checkpoint; never selected by filename glob or mtime")
    parser.add_argument("--experiment", required=True, help="Logical experiment identifier")
    parser.add_argument("--run-name", required=True, help="Actual training run directory name")
    parser.add_argument("--expected-iter", type=int, required=True,
                        help="Expected saved checkpoint iter, e.g. 499 for 500 learning iterations")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-exit-file", type=Path, required=True)
    parser.add_argument("--training-record", type=Path,
                        help="Optional schema_version=1 launcher record; checked if supplied")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.expected_iter < 0:
        parser.error("--expected-iter must be nonnegative")
    for field in ("experiment", "run_name"):
        value = getattr(args, field)
        if not value or value in {".", ".."} or not all(c.isalnum() or c in "_.-" for c in value):
            parser.error(f"--{field.replace('_', '-')} must be a safe identifier")
    for field in ("repo", "checkpoint", "out", "training_exit_file", "training_record"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    return args


def verify_sources(repo):
    actual = {name: sha256(Path(repo) / name) for name in SOURCE_SHA256}
    require(actual == SOURCE_SHA256, "Official source hash mismatch; refuse altered inference/config")
    return actual


def verify_checkpoint(path, expected_iter, torch):
    metadata = file_metadata(path)
    # The unchanged official build_runner also uses torch.load for this trusted file.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), "Checkpoint must be a dictionary")
    require(type(checkpoint.get("iter")) is int and checkpoint["iter"] == expected_iter,
            "Checkpoint saved iteration does not equal --expected-iter")
    tensor_counts = {}
    for key in ("actor_state_dict", "critic_state_dict"):
        state = checkpoint.get(key)
        require(isinstance(state, dict) and state, f"Missing or empty {key}")
        for name, value in state.items():
            require(torch.is_tensor(value), f"Non-tensor checkpoint weight: {key}.{name}")
            require(bool(torch.isfinite(value).all()), f"Nonfinite checkpoint weight: {key}.{name}")
        tensor_counts[key] = len(state)
    metadata.update(saved_iteration=checkpoint["iter"], keys=sorted(checkpoint),
                    finite_weight_tensor_counts=tensor_counts)
    require(sha256(path) == metadata["sha256"], "Checkpoint changed while being verified")
    return metadata


def training_evidence(args, checkpoint):
    exit_file = file_metadata(args.training_exit_file)
    require(args.training_exit_file.read_text(encoding="utf-8").strip() == "0",
            "Training exit file must contain 0 before evaluation")
    require(args.checkpoint.parent.name == args.run_name,
            "--run-name must match the chosen checkpoint's actual parent directory")
    evidence = {
        "training_exit_file": exit_file, "reported_training_exit_code": 0,
        "training_record_status": "NOT_SUPPLIED_TRAINING_UNVERIFIED",
        "training_record_consistency_verified": False,
        "training_reward": {"label": TRAINING_REWARD_LABEL, "tag": "Train/mean_reward",
                            "last": None, "source": None, "reward_scales": None},
        "caveat": "A supplied 0 exit file alone does not bind a process or prove training. "
                  "A launcher record provides file-bound provenance, not independent re-execution.",
    }
    if args.training_record is None:
        return evidence
    record_meta = file_metadata(args.training_record)
    record = json.loads(args.training_record.read_text(encoding="utf-8"))
    require(isinstance(record, dict) and record.get("schema_version") == 1,
            "Unsupported training record schema")
    for key, expected in (("experiment", args.experiment), ("run_name", args.run_name),
                          ("state", "completed"), ("exit_code", 0),
                          ("expected_final_iteration", args.expected_iter),
                          ("expected_iterations", args.expected_iter + 1),
                          ("source_commit", OFFICIAL_COMMIT), ("source_sha256", SOURCE_SHA256)):
        require(record.get(key) == expected, f"Training record mismatch: {key}")
    for key, expected in (("repo", args.repo), ("run_dir", args.checkpoint.parent),
                          ("training_exit_file", args.training_exit_file)):
        require(Path(record.get(key, "")).resolve() == expected,
                f"Training record path mismatch: {key}")
    recorded_checkpoint = record.get("checkpoint", {})
    require(Path(recorded_checkpoint.get("path", "")).resolve() == args.checkpoint,
            "Training record references another checkpoint")
    for key in ("sha256", "saved_iteration", "bytes"):
        require(recorded_checkpoint.get(key) == checkpoint[key],
                f"Training record checkpoint mismatch: {key}")
    config_path = Path(record["effective_config"])
    require(config_path.is_absolute(), "Training effective_config must be an absolute path")
    config_meta = file_metadata(config_path)
    require(record.get("effective_config_sha256") == config_meta["sha256"],
            "Training effective_config hash mismatch")
    effective = json.loads(config_path.read_text(encoding="utf-8"))
    require(isinstance(effective, dict), "Training effective_config must be an object")
    training_log = Path(record["training_log"])
    require(training_log.is_absolute(), "Training log path must be absolute")
    training_reward = record.get("training_reward", {})
    require(training_reward.get("tag") == "Train/mean_reward", "Unexpected training reward tag")
    last = training_reward.get("last")
    if last is not None:
        require(isinstance(last, dict) and math.isfinite(float(last["value"])),
                "Training reward must be finite")
        require(last.get("step") == args.expected_iter,
                "Training record reward is not from the expected final iteration")
    evidence.update({
        "training_record_status": "LAUNCHER_RECORD_CONSISTENCY_VERIFIED",
        "training_record_consistency_verified": True,
        "training_record_file": record_meta, "launcher_record": record,
        "effective_training_config_file": config_meta,
        "effective_training_config": effective,
        "training_log_file": file_metadata(training_log),
        "training_reward": {"label": TRAINING_REWARD_LABEL, "tag": "Train/mean_reward",
                            "last": last, "source": "supplied_launcher_training_record",
                            "reward_scales": training_scales(effective)},
    })
    return evidence


def training_scales(effective):
    """Retain full config even when a launcher uses an unrecognized wrapper key."""
    for candidate in (effective, effective.get("environment_effective", {}),
                      effective.get("environment_config", {}),
                      effective.get("env_config", {}), effective.get("env_cfg", {})):
        if isinstance(candidate, dict) and isinstance(candidate.get("reward_config"), dict):
            return candidate["reward_config"].get("scales")
    return None


def wilson(successes, total):
    require(total > 0 and 0 <= successes <= total, "Invalid binomial counts")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, midpoint - half), min(1.0, midpoint + half)]


def summarize(rows):
    require([row["reset_seed"] for row in rows] == list(SEEDS),
            "A complete evaluation must contain seeds 42--51 exactly once in order")
    successes = sum(row["success"] for row in rows)
    grasps = sum(row["grasped"] for row in rows)
    rewards = [row["common_default_evaluation_reward"] for row in rows]
    return {
        "episodes": len(rows), "seeds": list(SEEDS),
        "primary_metric": "success_rate", "successes": successes,
        "success_rate": successes / len(rows), "success_rate_wilson_95": wilson(successes, len(rows)),
        "secondary_metric": "grasp_rate", "grasps": grasps, "grasp_rate": grasps / len(rows),
        "grasp_rate_wilson_95": wilson(grasps, len(rows)),
        "evaluation_reward_label": EVALUATION_REWARD_LABEL,
        "mean_common_default_evaluation_reward": statistics.mean(rewards),
        "common_default_evaluation_reward_sample_std": statistics.stdev(rewards),
        "mean_final_distance_m": statistics.mean(row["final_distance_m"] for row in rows),
        "early_terminations": sum(row["terminated_early"] for row in rows),
    }


def save_video(env, trajectory, path):
    import imageio.v2 as iio
    import numpy as np
    fps = 1.0 / float(env.dt) / RENDER_EVERY
    frames_written = 0
    # Keep one official episode trajectory, at most ten rendered frames at a time,
    # and stream encoding. Never concatenate RGB frames across episodes.
    with iio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=16) as writer:
        for start in range(0, len(trajectory), RENDER_BATCH_STATES * RENDER_EVERY):
            states = trajectory[start:start + RENDER_BATCH_STATES * RENDER_EVERY:RENDER_EVERY]
            frames = env.render(states, height=480, width=640)  # Official default camera.
            require(len(frames) == len(states), "Renderer returned an unexpected frame count")
            for frame in frames:
                frame = np.asarray(frame)
                require(tuple(frame.shape) == (480, 640, 3), "Unexpected rendered image size")
                writer.append_data(frame)
                frames_written += 1
            del frames
    decoded_frames = 0
    # Actually decode every frame sequentially; count_frames alone is insufficient.
    with iio.get_reader(str(path)) as reader:
        for frame in reader:
            require(tuple(frame.shape) == (480, 640, 3), "Unexpected decoded image size")
            decoded_frames += 1
    require(decoded_frames == frames_written and decoded_frames > 0, "Incomplete video decoding")
    return {**file_metadata(path), "status": "DECODE_VERIFIED", "frames": frames_written,
            "decoded_frames": decoded_frames, "fps": fps, "width": 640, "height": 480,
            "render_every_steps": RENDER_EVERY, "camera": "official env.render default",
            "verified_decodable": True, "manual_visual_review": "NOT_PERFORMED"}


def evaluate_episodes(args, env, policy, cfg, official, jax, torch):
    jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
    is_dict_obs = isinstance(env.observation_size, dict)
    rows, video_errors = [], []
    video_dir = args.out / "videos"
    video_dir.mkdir()
    for episode, seed in enumerate(SEEDS, start=1):
        torch.manual_seed(seed)
        trajectory, reward, grasped, success, distance = official.rollout_single_episode(
            env, policy, jax.random.PRNGKey(seed), cfg.episode_length, is_dict_obs,
            jit_reset=jit_reset, jit_step=jit_step, cube_x=None, cube_y=None, obs_noise=0.0)
        require(len(trajectory) >= 2, "Official rollout returned no transition")
        max_z = max(official._cube_z(state) for state in trajectory)
        require(all(math.isfinite(float(value)) for value in (reward, distance, max_z)),
                "Nonfinite evaluation result")
        require(bool(success) == (float(distance) < 0.05), "Official success threshold mismatch")
        require(bool(grasped) == (max_z > 0.2), "Official grasp threshold mismatch")
        row = {
            "experiment": args.experiment, "run_name": args.run_name,
            "checkpoint_saved_iteration": args.expected_iter, "episode": episode,
            "reset_seed": seed, "torch_seed": seed,
            "evaluation_reward_label": EVALUATION_REWARD_LABEL,
            "common_default_evaluation_reward": float(reward), "final_distance_m": float(distance),
            "success": bool(success), "grasped": bool(grasped), "max_cube_z_m": float(max_z),
            "steps": len(trajectory) - 1, "terminated_early": len(trajectory) - 1 < cfg.episode_length,
            "initial_cube_xyz_m": official._box_pos(trajectory[0]).tolist(),
            "final_cube_xyz_m": official._box_pos(trajectory[-1]).tolist(),
            "target_xyz_m": official._target_pos(trajectory[0]).tolist(),
            "cube_x_override": None, "cube_y_override": None, "observation_noise": 0.0,
            "video": {"status": "NOT_REQUESTED", "verified_decodable": None},
        }
        if episode <= VIDEO_EPISODES:
            path = video_dir / f"episode_{episode:02d}_seed_{seed}.mp4"
            try:
                row["video"] = save_video(env, trajectory, path)
            except Exception as error:
                row["video"] = {"path": str(path), "status": "FAILED", "verified_decodable": False,
                                "error": f"{type(error).__name__}: {error}"}
                video_errors.append({"episode": episode, **row["video"]})
        rows.append(row)
        with (args.out / "episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print("EPISODE", json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        del trajectory
    return rows, video_errors


def installed_versions():
    versions = {"python": platform.python_version()}
    for package in ("torch", "jax", "jaxlib", "numpy", "mujoco", "mujoco-mjx", "playground",
                    "rsl-rl-lib", "imageio", "imageio-ffmpeg", "tensorboard"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def artifact_manifest(out):
    return [{"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(out.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json"]


def main(argv=None):
    args = parse_args(argv)
    require(not args.out.exists() or (args.out.is_dir() and not any(args.out.iterdir())),
            "Output directory must be absent or empty; never mix evaluation attempts")
    sources = verify_sources(args.repo)
    require(args.training_exit_file.read_text(encoding="utf-8").strip() == "0",
            "Training exit file must contain 0 before importing GPU packages")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(args.repo))
    import torch
    checkpoint = verify_checkpoint(args.checkpoint, args.expected_iter, torch)
    training = training_evidence(args, checkpoint)
    import jax
    import inference as official
    from mujoco_playground import registry
    from device_utils import apply_mjx_overrides_to_playground_cfg
    require(Path(official.__file__).resolve() == args.repo / "inference.py", "Wrong inference module imported")
    require(official.SUCCESS_THRESHOLD == 0.05 and official.GRASP_Z_THRESHOLD == 0.2,
            "Unexpected official metric thresholds")
    cfg = registry.get_default_config(official.ENV_NAME)
    default_cfg = cfg.to_dict()
    require(default_cfg.get("reward_config", {}).get("scales") == DEFAULT_REWARD_SCALES,
            "Installed Playground default reward changed; cross-experiment comparison requires audit")
    apply_mjx_overrides_to_playground_cfg(cfg)
    effective_cfg = cfg.to_dict()
    require(effective_cfg.get("reward_config") == default_cfg["reward_config"],
            "Device overrides changed default evaluation reward")
    env = registry.load(official.ENV_NAME, config=cfg)
    require(env._config.to_dict().get("reward_config") == default_cfg["reward_config"],
            "Loaded environment changed default evaluation reward")
    runner = official.build_runner(str(args.checkpoint), args.device)
    policy = runner.get_inference_policy(device=args.device)
    environment_source = Path(inspect.getfile(type(env))).resolve()
    metadata = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment, "run_name": args.run_name, "checkpoint": checkpoint,
        "expected_saved_iteration": args.expected_iter,
        "official_source_reference_commit": OFFICIAL_COMMIT, "official_source_sha256": sources,
        "source_verification": "Exact file fingerprints from the pinned official commit; no Git HEAD claim",
        "evaluator": file_metadata(Path(__file__)), "environment_source": file_metadata(environment_source),
        "versions": installed_versions(), "platform": platform.platform(),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "torch_device": args.device, "jax_devices": [str(device) for device in jax.devices()],
        "runtime_environment": {key: os.environ.get(key) for key in
                                ("MUJOCO_GL", "XLA_PYTHON_CLIENT_PREALLOCATE", "WANDB_MODE", "CUDA_VISIBLE_DEVICES")},
        "environment_name": official.ENV_NAME, "environment_config": effective_cfg,
        "default_environment_config": default_cfg, "evaluation_reward_scales": DEFAULT_REWARD_SCALES,
        "evaluation_reward_label": EVALUATION_REWARD_LABEL,
        "training_reward_label": TRAINING_REWARD_LABEL,
        "reward_comparison_warning": "Training reward and common-default evaluation return are distinct. "
                                     "Changed training reward weights make raw training returns incomparable.",
        "seeds": list(SEEDS), "episodes": len(SEEDS), "torch_reseed_per_episode": True,
        "rollout": "unchanged official inference.rollout_single_episode",
        "policy": "official deterministic get_inference_policy; unchanged clipping to [-1, 1]",
        "primary_metric": "success_rate", "success_definition": "final cube-to-target Euclidean distance < 0.05 m",
        "secondary_metric": "grasp_rate", "grasp_definition": "maximum cube z over episode > 0.2 m",
        "episode_length": int(cfg.episode_length), "control_dt_s": float(env.dt),
        "condition": {"cube_x": None, "cube_y": None, "obs_noise": 0.0},
        "videos": {"episodes": [1, 2, 3], "seeds": [42, 43, 44], "width": 640, "height": 480,
                   "render_every_steps": RENDER_EVERY, "render_batch_states": RENDER_BATCH_STATES,
                   "camera": "official env.render default", "manual_visual_review": "NOT_PERFORMED"},
        "small_sample_caveat": "Ten paired episodes are a coarse empirical comparison, not a population guarantee.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "metadata.json", metadata)
    write_json(args.out / "training_evidence.json", training)
    write_json(args.out / "summary.json", {"status": "RUNNING", "experiment": args.experiment,
                                           "run_name": args.run_name})
    started = time.monotonic()
    try:
        rows, video_errors = evaluate_episodes(args, env, policy, cfg, official, jax, torch)
        require(sha256(args.checkpoint) == checkpoint["sha256"], "Checkpoint changed during evaluation")
        require(verify_sources(args.repo) == sources, "Course sources changed during evaluation")
        summary = {
            "status": "METRICS_COMPLETE_VIDEO_FAILED" if video_errors else "METRICS_AND_VIDEO_FILES_COMPLETE",
            "experiment": args.experiment, "run_name": args.run_name, "checkpoint": checkpoint,
            **summarize(rows), "training_reward": training["training_reward"],
            "training_record_status": training["training_record_status"], "video_errors": video_errors,
            "video_decoding_verified": not video_errors, "manual_visual_review": "NOT_PERFORMED",
            "elapsed_seconds": time.monotonic() - started,
        }
        fields = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
        with (args.out / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        write_json(args.out / "summary.json", summary)
        write_json(args.out / "artifact_manifest.json", artifact_manifest(args.out))
        print("P2_EVALUATION_COMPLETE", json.dumps(summary, allow_nan=False), flush=True)
        return 2 if video_errors else 0
    except Exception as error:
        write_json(args.out / "summary.json", {
            "status": "FAILED", "experiment": args.experiment, "run_name": args.run_name,
            "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic() - started,
            "manual_visual_review": "NOT_PERFORMED",
        })
        write_json(args.out / "artifact_manifest.json", artifact_manifest(args.out))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
