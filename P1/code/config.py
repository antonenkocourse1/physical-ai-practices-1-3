"""Общий конфиг среды и RSL-RL для PandaPickCube.

Импортируется из train.py и inference.py — изменения здесь
автоматически применяются к обоим скриптам.
"""

from mujoco_playground.config import manipulation_params

ENV_NAME = "PandaPickCube"

# Гиперпараметры сети (меняйте здесь)
ACTOR_HIDDEN_DIMS  = [512, 256, 128]
CRITIC_HIDDEN_DIMS = [512, 256, 128]
ACTIVATION         = "elu"
INIT_NOISE_STD     = 1.0


def build_runner_cfg(obs_size) -> dict:
    """Строит конфиг OnPolicyRunner для RSL-RL 3.x.

    Args:
        obs_size: raw_env.observation_size — int или dict (асимметричные obs).

    Returns:
        Словарь, готовый для передачи в OnPolicyRunner(..., train_cfg=...).
    """
    cfg = manipulation_params.rsl_rl_config(ENV_NAME).to_dict()

    # Конвертируем старый формат playground (ключ «policy») в RSL-RL 3.x
    # (отдельные ключи «actor» и «critic» с полем class_name).
    print(f"{cfg=}")
    policy = cfg.pop("policy", {})
    cfg["actor"] = {
        "class_name": "rsl_rl.models.mlp_model.MLPModel",
        "hidden_dims": ACTOR_HIDDEN_DIMS,
        "activation": ACTIVATION,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": INIT_NOISE_STD,
        },
    }
    cfg["critic"] = {
        "class_name": "rsl_rl.models.mlp_model.MLPModel",
        "hidden_dims": CRITIC_HIDDEN_DIMS,
        "activation": ACTIVATION,
    }

    if isinstance(obs_size, dict):
        cfg["obs_groups"] = {"actor": ["state"], "critic": ["privileged_state"]}
    else:
        cfg["obs_groups"] = {"actor": ["state"], "critic": ["state"]}
    print(f"{cfg=}")
    return cfg
