from pettingzoo.test import  performance_benchmark, max_cycles_test
from petri_env import petri_energy_env
from rendering import render_test
import json
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
import yaml
import pickle
import random
import torch

def my_app(cfg, actions) -> None:

    env_config = cfg["env_config"]
    random.seed(env_config["seed_val"])
    np.random.seed(env_config["seed_val"])
    torch.manual_seed(env_config["seed_val"])

    env = petri_energy_env.env(env_config, None)

    render_test(env, actions)

if __name__ == "__main__":
    path = "/Users/grigoriiguz/projects/bacterias/outputs/2021-12-01/full_0_old"
    with open(os.path.join(path, ".hydra/config.yaml"), "r") as stream:
        config = yaml.safe_load(stream=stream)

    with open(os.path.join(path, "env_store.npy"), "rb") as stream:
        _, actions = pickle.load(stream)
    my_app(config, actions)
