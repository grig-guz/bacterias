import random
import numpy as np
import torch


from pettingzoo.test import  performance_benchmark, max_cycles_test
from petri_env import petri_energy_env
from rendering import render_test
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
sys.setrecursionlimit(100000)


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    env_config = cfg.env_config
    random.seed(env_config["seed_val"])
    np.random.seed(env_config["seed_val"])
    torch.manual_seed(env_config["seed_val"])
    neat_config = None
    env = petri_energy_env.env(env_config, neat_config)

    render_test(env)

    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
