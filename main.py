from pettingzoo.test import  performance_benchmark, max_cycles_test
from petri_env import petri_env
from rendering import render_test
import json
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:

    env_config = cfg.env_config
    env = petri_env.env(env_config)

    render_test(env)

    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
