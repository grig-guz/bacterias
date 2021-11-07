from pettingzoo.test import render_test, performance_benchmark, max_cycles_test
from petri_env import petri_env


env = petri_env.env()

render_test(env)

