from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from petri_env.petri_scenario import PetriScenario


class raw_env(SimpleEnv):
    def __init__(self, max_cycles=500000, continuous_actions=False):
        scenario = PetriScenario()
        world = scenario.make_world()
        super().__init__(scenario, world, max_cycles, continuous_actions)
        self.metadata['name'] = "petri_env"

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
