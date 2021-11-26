import random
import math
import numpy as np
import torch
from pettingzoo.mpe._mpe_utils.rendering import FilledPolygon, PolyLine

def collect_render_results(env, mode):
    results = []

    env.reset()
    for i in range(1000000):
        if i > 0:
            for agent in env.agent_iter(env.num_agents // 2 + 1):
                obs, reward, done, info = env.last()
                if done:
                    action = None
                else:
                    unwrapped = env.unwrapped
                    with torch.no_grad():
                        action = unwrapped.world.agents[unwrapped._index_map[agent]].policy(obs)
                env.step(action)
        if i % 20 == 0:
            render_result = env.render(mode=mode)
        #results.append(render_result)

    return results


def render_test(env, custom_tests={}):
    render_modes = env.metadata.get('render.modes')
    assert render_modes is not None, "Environment's that support rendering must define render modes in metadata"
    for mode in render_modes:
        render_results = collect_render_results(env, mode)
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
    sq2 = radius * math.sqrt(2)
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
