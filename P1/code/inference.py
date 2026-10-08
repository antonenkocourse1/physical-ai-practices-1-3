"""Инференс обученной политики PandaPickCube.

Запуск:
    python3 inference.py --checkpoint logs/PandaPickCube_20240322_143025/model_300.pt
    python3 inference.py --checkpoint logs/.../model_300.pt --record video.mp4
    python3 inference.py --checkpoint logs/.../model_300.pt --episodes 10
"""

import argparse
import os

from device_utils import (
    apply_mjx_overrides_to_playground_cfg,
    default_torch_device,
    rslrl_brax_wrapper_jax_device_rank,
)

import jax
import mujoco
import mujoco.viewer
import numpy as np
import torch
from mujoco_playground import registry, wrapper_torch
from rsl_rl.runners import OnPolicyRunner

from config import ENV_NAME, build_runner_cfg


jax.config.update("jax_default_matmul_precision", "highest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Инференс обученной политики PandaPickCube"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к чекпоинту model_N.pt")
    parser.add_argument("--device", type=str, default=None,
                        help="Устройство PyTorch (по умолчанию: лучший доступный cuda/mps/cpu)")
    parser.add_argument("--record", type=str, default=None,
                        help="Путь для сохранения видео (mp4). "
                             "По умолчанию сохраняется рядом с чекпоинтом.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Количество эпизодов (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--cube_x", type=float, default=None,
                        help="Начальная позиция куба по оси X (по умолчанию — случайная)")
    parser.add_argument("--cube_y", type=float, default=None,
                        help="Начальная позиция куба по оси Y (по умолчанию — случайная)")
    parser.add_argument("--obs_noise", type=float, default=0.0,
                        help="Стандартное отклонение гауссовского шума, "
                             "добавляемого к наблюдениям на каждом шаге "
                             "(имитация шума сенсоров; по умолчанию 0.0 — без шума)")
    return parser.parse_args()


# Порог по z-координате куба: если в течение эпизода куб поднялся выше,
# считаем, что робот его успешно схватил (оторвал от стола).
GRASP_Z_THRESHOLD = 0.2

# Порог по евклидову расстоянию между кубом и целью: если в конце эпизода
# расстояние меньше порога, эпизод засчитывается как успешный (success rate).
SUCCESS_THRESHOLD = 0.05


def _maybe_add_obs_noise(obs_torch: torch.Tensor, obs_noise: float) -> torch.Tensor:
    if obs_noise > 0.0:
        obs_torch = obs_torch + torch.randn_like(obs_torch) * obs_noise
    return obs_torch


def _cube_z(state) -> float:
    return float(state.data.qpos[..., 11])


def _box_pos(state) -> np.ndarray:
    # Куб — freejoint после 9 DOF руки (7 arm + 2 gripper);
    # qpos[9:12] — это его x, y, z.
    qpos = state.data.qpos
    return np.array([
        float(qpos[..., 9]),
        float(qpos[..., 10]),
        float(qpos[..., 11]),
    ])


def _target_pos(state) -> np.ndarray:
    return np.array(state.info["target_pos"]).flatten()


def _pos_err(state) -> float:
    return float(np.linalg.norm(_target_pos(state) - _box_pos(state)))


def build_runner(checkpoint_path: str, device: str) -> OnPolicyRunner:
    """Создаёт OnPolicyRunner и загружает чекпоинт."""
    device_rank = rslrl_brax_wrapper_jax_device_rank(device)

    env_cfg = registry.get_default_config(ENV_NAME)
    apply_mjx_overrides_to_playground_cfg(env_cfg)
    raw_env = registry.load(ENV_NAME, config=env_cfg)

    brax_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env,
        num_actors=1,
        seed=1,
        episode_length=env_cfg.episode_length,
        action_repeat=1,
        device_rank=device_rank,
    )
    brax_env.device = torch.device(device)
    # RSL-RL 3.x Logger обращается к env.cfg, которого нет в RSLRLBraxWrapper
    brax_env.cfg = {}

    cfg_dict = build_runner_cfg(raw_env.observation_size)

    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    runner = OnPolicyRunner(brax_env, cfg_dict, checkpoint_dir, device=device)
    # Чекпоинт с Colab/GPU хранит тензоры на CUDA; без map_location torch.load падает на CPU/MPS.
    runner.load(checkpoint_path, map_location=device)
    return runner


def _override_cube_pos(state, cube_x, cube_y):
    """Переопределяет начальную позицию куба в состоянии среды.

    Куб — freejoint после 9 DOF руки (7 arm + 2 gripper).
    qpos индексы: 9 = x, 10 = y.
    """
    qpos = state.data.qpos
    if cube_x is not None:
        qpos = qpos.at[..., 9].set(cube_x)
    if cube_y is not None:
        qpos = qpos.at[..., 10].set(cube_y)
    data = state.data.replace(qpos=qpos)
    return state.replace(data=data)


def rollout_single_episode(env, policy, rng, episode_length, is_dict_obs,
                           jit_reset, jit_step,
                           cube_x=None, cube_y=None, obs_noise=0.0):
    """Прогоняет один эпизод и возвращает
    (trajectory, total_reward, grasped, success, final_pos_err)."""
    state = jit_reset(rng)
    if cube_x is not None or cube_y is not None:
        state = _override_cube_pos(state, cube_x, cube_y)

    trajectory = [state]
    total_reward = 0.0
    max_cube_z = _cube_z(state)
    final_pos_err = _pos_err(state)

    for _ in range(episode_length):
        obs = state.obs["state"] if is_dict_obs else state.obs
        obs_torch = wrapper_torch._jax_to_torch(obs)
        obs_torch = _maybe_add_obs_noise(obs_torch, obs_noise)

        with torch.no_grad():
            actions = policy({"state": obs_torch})
            actions = torch.clip(actions, -1.0, 1.0)

        state = jit_step(state, wrapper_torch._torch_to_jax(actions.flatten()))
        trajectory.append(state)
        total_reward += float(state.reward)
        max_cube_z = max(max_cube_z, _cube_z(state))
        final_pos_err = _pos_err(state)

        if state.done:
            break

    grasped = max_cube_z > GRASP_Z_THRESHOLD
    success = final_pos_err < SUCCESS_THRESHOLD
    return trajectory, total_reward, grasped, success, final_pos_err


def run_interactive(env, policy, args, env_cfg, is_dict_obs):
    """Интерактивный режим — рендер через mujoco.viewer (видно через VNC)."""
    import time

    mj_model = env.mj_model

    print("Запуск интерактивного просмотра через mujoco.viewer...")
    print("(видно через VNC: http://localhost:6080/vnc.html)")
    print()

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    mj_data = mujoco.MjData(mj_model)
    grasps = 0
    successes = 0
    total_rewards = []
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for ep in range(args.episodes):
            rng = jax.random.PRNGKey(args.seed + ep)
            state = jit_reset(rng)
            if args.cube_x is not None or args.cube_y is not None:
                state = _override_cube_pos(state, args.cube_x, args.cube_y)
            total_reward = 0.0
            max_cube_z = _cube_z(state)
            final_pos_err = _pos_err(state)

            obs = state.obs["state"] if is_dict_obs else state.obs
            obs_torch = wrapper_torch._jax_to_torch(obs)
            obs_torch = _maybe_add_obs_noise(obs_torch, args.obs_noise)

            for _ in range(env_cfg.episode_length):
                if not viewer.is_running():
                    print("Viewer закрыт.")
                    return

                with torch.no_grad():
                    actions = policy({"state": obs_torch})
                    actions = torch.clip(actions, -1.0, 1.0)

                state = jit_step(
                    state, wrapper_torch._torch_to_jax(actions.flatten())
                )
                total_reward += float(state.reward)
                max_cube_z = max(max_cube_z, _cube_z(state))
                final_pos_err = _pos_err(state)

                mj_data.qpos[:] = np.array(state.data.qpos).flatten()
                mj_data.qvel[:] = np.array(state.data.qvel).flatten()
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

                time.sleep(env.dt)

                obs = state.obs["state"] if is_dict_obs else state.obs
                obs_torch = wrapper_torch._jax_to_torch(obs)
                obs_torch = _maybe_add_obs_noise(obs_torch, args.obs_noise)

                if state.done:
                    break

            grasped = max_cube_z > GRASP_Z_THRESHOLD
            success = final_pos_err < SUCCESS_THRESHOLD
            grasps += int(grasped)
            successes += int(success)
            total_rewards.append(total_reward)
            grasp_mark = "✔" if grasped else "✘"
            success_mark = "✔" if success else "✘"
            print(f"  Эпизод {ep + 1}/{args.episodes}: "
                  f"reward = {total_reward:.2f}, захват: {grasp_mark}, "
                  f"успех: {success_mark} (расстояние до цели = {final_pos_err:.3f} м, threshold = {SUCCESS_THRESHOLD:.2f} м)")

    if total_rewards:
        n = len(total_rewards)
        mean_reward = sum(total_rewards) / n
        print(f"\nИтого: захватов {grasps}/{n}, "
              f"success rate = {successes}/{n}, "
              f"средняя награда {mean_reward:.2f}")


def run_record(env, policy, args, env_cfg, is_dict_obs, record_path: str):
    """Запись видео в mp4 через imageio."""
    import imageio.v3 as iio

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    all_frames = []
    render_every = 2
    fps = 1.0 / env.dt / render_every

    grasps = 0
    successes = 0
    total_rewards = []
    for ep in range(args.episodes):
        rng = jax.random.PRNGKey(args.seed + ep)
        trajectory, total_reward, grasped, success, final_pos_err = rollout_single_episode(
            env, policy, rng, env_cfg.episode_length, is_dict_obs,
            jit_reset=jit_reset, jit_step=jit_step,
            cube_x=args.cube_x, cube_y=args.cube_y,
            obs_noise=args.obs_noise,
        )
        grasps += int(grasped)
        successes += int(success)
        total_rewards.append(total_reward)
        grasp_mark = "✔" if grasped else "✘"
        success_mark = "✔" if success else "✘"
        print(f"  Эпизод {ep + 1}/{args.episodes}: "
              f"reward = {total_reward:.2f}, захват: {grasp_mark}, "
              f"успех: {success_mark} (расстояние до цели = {final_pos_err:.3f} м, threshold = {SUCCESS_THRESHOLD:.2f} м)")

        traj_subset = trajectory[::render_every]
        frames = env.render(traj_subset, height=480, width=640)
        all_frames.extend(frames)

    n = len(total_rewards)
    mean_reward = sum(total_rewards) / n
    print(f"\nИтого: захватов {grasps}/{n}, "
          f"success rate = {successes}/{n}, "
          f"средняя награда {mean_reward:.2f}")
    print(f"\nСохранение видео: {record_path} ({len(all_frames)} кадров, {fps:.0f} fps)")
    os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
    iio.imwrite(record_path, np.stack(all_frames), fps=fps)
    print("Готово!")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = default_torch_device(args.device)

    print(f"Загрузка чекпоинта: {args.checkpoint}")
    runner = build_runner(args.checkpoint, device)
    policy = runner.get_inference_policy(device=device)

    env_cfg = registry.get_default_config(ENV_NAME)
    apply_mjx_overrides_to_playground_cfg(env_cfg)
    env = registry.load(ENV_NAME, config=env_cfg)
    is_dict_obs = isinstance(env.observation_size, dict)

    print(f"Среда:    {ENV_NAME}")
    print(f"Эпизодов: {args.episodes}")
    print()

    if args.record is not None:
        run_record(env, policy, args, env_cfg, is_dict_obs, args.record)
    else:
        run_interactive(env, policy, args, env_cfg, is_dict_obs)


if __name__ == "__main__":
    main()
