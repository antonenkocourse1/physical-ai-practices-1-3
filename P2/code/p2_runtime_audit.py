"""Read-only instrumentation imported by audited per-experiment train copies.

Nothing here installs packages, seeds RNGs, changes rewards/configs, or edits the
course sources. Training only happens in the original train main().
"""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

_CONTEXT = {}
DEFAULT_WEIGHTS = {"gripper_box": 4.0, "box_target": 8.0,
                   "no_floor_collision": 0.25, "robot_target_qpos": 0.3}
EXPECTED_VERSIONS = {"torch": "2.8.0", "jax": "0.6.2", "jaxlib": "0.6.2",
                     "mujoco": "3.5.0", "mujoco-mjx": "3.5.0",
                     "playground": "0.1.0", "rsl-rl-lib": "5.5.1"}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")
    tmp.replace(path)


def snapshot(value):
    return copy.deepcopy(value.to_dict() if hasattr(value, "to_dict") else value)


def initialize(args, log_dir, device, num_envs):
    audit = Path(os.environ["P2_AUDIT_DIR"])
    stage = json.loads((audit / "stage.json").read_text())
    spec = stage["spec"]
    assert args.resume is None, "P2 experiments must start fresh"
    assert args.exp_name == spec["run_name"]
    assert args.max_iters == spec["max_iters"] and num_envs == spec["num_envs"]
    assert args.seed == 1 and args.save_interval == 50 and device == "cuda:0"
    assert Path(log_dir).resolve() == Path(stage["run_dir"])
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    for name, expected in EXPECTED_VERSIONS.items():
        assert versions[name].split("+")[0] == expected, (name, versions[name], expected)
    import torch
    import jax
    assert torch.cuda.is_available() and torch.version.cuda == "12.8"
    assert torch.cuda.device_count() == 1 and jax.devices()[0].platform == "gpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    _CONTEXT.update(audit=audit, stage=stage, spec=spec,
                    capture={"created_utc": datetime.now(timezone.utc).isoformat(),
                      "kind": "live_pre_training_capture", "arguments": vars(args),
                      "num_envs": num_envs, "device": device, "versions": versions,
                      "torch_cuda": torch.version.cuda,
                      "torch_cudnn": torch.backends.cudnn.version(),
                      "gpu": torch.cuda.get_device_name(0),
                      "jax_devices": [str(d) for d in jax.devices()],
                      "environment_variables": {name: os.environ.get(name) for name in
                          ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE",
                           "JAX_COMPILATION_CACHE_DIR", "MUJOCO_GL", "WANDB_MODE")},
                      "torch_initial_seed_observed_not_set": torch.initial_seed(),
                      "seed_caveat": "Official behavior preserved: environment seed 1; Torch initialization is not explicitly seeded."})
    write_json(audit / "runtime_identity.json", _CONTEXT["capture"])


def capture_environment(env_cfg, label):
    value = snapshot(env_cfg)
    weights = value["reward_config"]["scales"]
    if label == "default":
        assert weights == DEFAULT_WEIGHTS, f"Unexpected default weights: {weights}"
    else:
        assert label == "effective"
        wanted = dict(DEFAULT_WEIGHTS)
        change = _CONTEXT["spec"]["reward_override"]
        if change:
            wanted[change["name"]] = change["value"]
        assert weights == wanted, f"Reward isolation failed: {weights}, expected {wanted}"
        original = copy.deepcopy(_CONTEXT["capture"]["environment_default"])
        if change:
            original["reward_config"]["scales"][change["name"]] = change["value"]
        assert value == original, "An unexpected environment parameter changed"
    _CONTEXT["capture"]["environment_" + label] = value
    write_json(_CONTEXT["audit"] / ("environment_" + label + ".json"), value)


def capture_runner(cfg_dict, raw_env):
    cfg = snapshot(cfg_dict)
    for role in ("actor", "critic"):
        assert cfg[role]["hidden_dims"] == [512, 256, 128]
        assert cfg[role]["activation"] == "elu"
    assert cfg["actor"]["distribution_cfg"]["init_std"] == 1.0
    assert cfg["seed"] == 1 and cfg["save_interval"] == 50
    assert cfg["max_iterations"] == _CONTEXT["spec"]["max_iters"]
    cap = _CONTEXT["capture"]
    cap.update(runner_effective_pre_constructor=cfg,
               observation_size=raw_env.observation_size,
               action_size=raw_env.action_size,
               experience_per_iteration=_CONTEXT["spec"]["num_envs"] * cfg["num_steps_per_env"],
               expected_total_environment_steps=_CONTEXT["spec"]["num_envs"] * cfg["num_steps_per_env"] * _CONTEXT["spec"]["max_iters"])
    # Constructor destructively pops actor/critic entries: this copy precedes it.
    write_json(_CONTEXT["audit"] / "effective_config.json", cap)


def capture_elapsed(seconds):
    write_json(_CONTEXT["audit"] / "learn_timing.json", {
        "training_learn_seconds": seconds,
        "scope": "Same original t_start immediately before runner.learn through return, includes first learn compilation, excludes runner construction and evaluator.",
        "finished_utc": datetime.now(timezone.utc).isoformat()})
