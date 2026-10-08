"""Интерактивное управление Franka Panda через MuJoCo GUI.

Сцена берётся из MuJoCo Playground (как в train/inference), а не из пути к menagerie:
в wheel пакета нет ``external_deps/mujoco_menagerie/...`` на диске.

Запуск (внутри контейнера):
    python3 panda_teleop.py

В окне viewer'а:
    Ctrl+M  -- показать/скрыть слайдеры управления суставами
    Ctrl+R  -- сбросить в исходную позу
    Пробел  -- пауза физики
    Esc     -- выход
"""

import signal
import sys
import time

from device_utils import apply_mjx_overrides_to_playground_cfg

import jax
import mujoco
import mujoco.viewer
from mujoco_playground import registry

from config import ENV_NAME

signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

jax.config.update("jax_default_matmul_precision", "highest")

env_cfg = registry.get_default_config(ENV_NAME)
apply_mjx_overrides_to_playground_cfg(env_cfg)
_env = registry.load(ENV_NAME, config=env_cfg)

model = _env.mj_model
data = mujoco.MjData(model)
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)

print(f"Сцена «{ENV_NAME}» (MuJoCo Playground).")
print("  Ctrl+M  — слайдеры управления суставами руки")
print("  Ctrl+R  — сброс позы")
print("  Пробел  — пауза")
print("  Esc     — выход")
print()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
