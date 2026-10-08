"""Обучение Panda PickCube с RSL-RL (PPO).

Запуск:
    python3 train.py
    python3 train.py --num_envs 4096 --max_iters 1001
    python3 train.py --exp_name my_run --resume logs/my_run/model_150.pt
"""

import argparse
import json
import os
import time
from datetime import datetime

from device_utils import (
    apply_mjx_overrides_to_playground_cfg,
    default_num_envs,
    default_torch_device,
    rslrl_brax_wrapper_jax_device_rank,
)

import jax
import torch
from mujoco_playground import registry, wrapper_torch
from rsl_rl.runners import OnPolicyRunner

from config import ENV_NAME, build_runner_cfg


jax.config.update("jax_default_matmul_precision", "highest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение PandaPickCube (RSL-RL PPO)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Имя эксперимента (default: PandaPickCube_YYYYMMDD_HHMMSS)")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Число параллельных сред (по умолчанию: 4096 на CUDA, меньше на CPU/MPS)")
    parser.add_argument("--max_iters", type=int, default=1001,
                        help="Количество итераций обучения (default: 1001)")
    parser.add_argument("--device", type=str, default=None,
                        help="Устройство PyTorch (по умолчанию: лучший доступный cuda/mps/cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Путь к чекпоинту для продолжения обучения")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Интервал сохранения чекпоинтов (default: 50)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    return parser.parse_args()


def resolve_log_dir(args) -> str:
    """Возвращает директорию лога и создаёт её."""
    if args.exp_name:
        exp_name = args.exp_name
    elif args.resume:
        # Продолжаем тот же эксперимент: берём имя из пути к чекпоинту
        exp_name = os.path.basename(os.path.dirname(os.path.abspath(args.resume)))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{ENV_NAME}_{timestamp}"

    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def main():
    args = parse_args()

    device = default_torch_device(args.device)
    num_envs = default_num_envs(args.num_envs)
    device_rank = rslrl_brax_wrapper_jax_device_rank(device)

    log_dir = resolve_log_dir(args)

    print(f"Среда:            {ENV_NAME}")
    print(f"Устройство:       {device}")
    print(f"Параллельных сред: {num_envs}")
    print(f"Итерации:         {args.max_iters}")
    print(f"Лог/чекпоинты:    {log_dir}")
    print()

    env_cfg = registry.get_default_config(ENV_NAME)
    apply_mjx_overrides_to_playground_cfg(env_cfg)
    raw_env = registry.load(ENV_NAME, config=env_cfg)

    brax_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env,
        num_actors=num_envs,
        seed=args.seed,
        episode_length=env_cfg.episode_length,
        action_repeat=1,
        device_rank=device_rank,
    )
    brax_env.device = torch.device(device)
    # RSL-RL 3.x Logger обращается к env.cfg, которого нет в RSLRLBraxWrapper
    brax_env.cfg = {}

    cfg_dict = build_runner_cfg(raw_env.observation_size)
    cfg_dict["seed"] = args.seed
    cfg_dict["max_iterations"] = args.max_iters
    cfg_dict["save_interval"] = args.save_interval

    # Сохраняем конфиг до создания runner'а (construct_algorithm делает pop на actor/critic)
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)
    print(f"Конфиг сохранён: {config_path}")

    runner = OnPolicyRunner(brax_env, cfg_dict, log_dir, device=device)

    if args.resume:
        print(f"Загрузка чекпоинта: {args.resume}")
        runner.load(args.resume, map_location=device)

    print("=" * 60)
    print("  Начинаю обучение...")
    print("=" * 60)
    print()

    t_start = time.time()
    runner.learn(
        num_learning_iterations=args.max_iters,
        init_at_random_ep_len=False,
    )
    elapsed = time.time() - t_start

    print()
    print("=" * 60)
    print(f"  Обучение завершено за {elapsed:.0f} с ({elapsed / 60:.1f} мин)")
    print(f"  Логи и чекпоинты: {log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
