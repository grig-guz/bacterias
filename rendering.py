import random
import math
import numpy as np
import torch
from pettingzoo.mpe._mpe_utils.rendering import FilledPolygon, PolyLine
import os
import pickle

def collect_render_results(env, mode):
    print("Working directory : {}".format(os.getcwd()))
    results = []
    env.reset()
    actions_buffer = []
    for i in range(300000):
        for agent in env.agent_iter(env.num_agents):
            obs, reward, done, info = env.last()
            if done:
                action = None
            else:
                unwrapped = env.unwrapped
                with torch.no_grad():
                    policy = unwrapped.world.agents[unwrapped._index_map[agent]].policy
                    if isinstance(policy, torch.nn.Module):
                        action = unwrapped.world.agents[unwrapped._index_map[agent]].policy(obs)
                    else:
                        action = policy.activate(obs)
                        action = (np.argmax(action[:4]), np.argmax(action[4:]))
                        
            env.step(action)
            actions_buffer.append(action)
        if i % 10 == 0:
            render_result = env.render(mode=mode)
            results.append(render_result)
    actions_buffer = np.array(actions_buffer)
    agent_types = np.array([np.concatenate([agent.consumes, agent.color, agent.produces]) for agent in env.unwrapped.world.agents])
    print(len(actions_buffer))
    with open(os.path.join(os.getcwd(), "env_store.npy"), "wb") as f:
        np.save(f, actions_buffer)
        np.save(f, agent_types)

    return results

def collect_saved_render_results(env, actions, mode):
    print("Working directory : {}".format(os.getcwd()))
    results = []
    env.reset()
    idx = 0
    step = 0
    while idx < len(actions):
        for _ in env.agent_iter(env.num_agents):
            _, _, _, _ = env.last()
            action = actions[idx]
            env.step(action)
            idx += 1
        
        step += 1
        if step % 5 == 0:
            render_result = env.render(mode=mode)
            results.append(render_result)
    return results



def render_test(env, actions=None, custom_tests={}):
    render_modes = env.metadata.get('render.modes')
    assert render_modes is not None, "Environment's that support rendering must define render modes in metadata"
    render_modes = ["human"]
    for mode in render_modes:
        if actions is None:
            render_results = collect_render_results(env, mode)
        else:
            render_results = collect_saved_render_results(env, actions, mode)
        for res in render_results:
            if mode in custom_tests.keys():
                assert custom_tests[mode](res)
            if mode == 'rgb_array':
                assert isinstance(res, np.ndarray) and len(res.shape) == 3 and res.shape[2] == 3 and res.dtype == np.uint8, f"rgb_array mode must return a valid image array, is {res}"
            if mode == 'ansi':
                assert isinstance(res, str)  # and len(res.shape) == 3 and res.shape[2] == 3 and res.dtype == np.uint8, "rgb_array mode must have shit in it"
            if mode == "human":
                assert res is None
    env.close()


def make_star(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)

def make_square(radius=10, res=30, filled=True):
    points = []
    points.append((radius, radius))
    points.append((radius, 0))
    points.append((radius, -radius))
    points.append((0, -radius))
    points.append((-radius, -radius))
    points.append((-radius, 0))
    points.append((-radius, radius))
    points.append((0, radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)

def make_triangle(radius=10, res=30, filled=True):
    points = []
    points.append((0, radius))
    points.append((radius, -radius))
    points.append((-radius, -radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)
