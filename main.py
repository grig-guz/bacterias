from pettingzoo.test import  performance_benchmark, max_cycles_test
from petri_env import petri_env
from rendering import render_test

env = petri_env.env()

render_test(env)

