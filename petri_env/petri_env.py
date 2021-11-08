from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils import rendering
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.agent_selector import agent_selector
from petri_env.petri_scenario import PetriScenario
import numpy as np
from gym import spaces
import copy


class raw_env(SimpleEnv):
    def __init__(self, max_cycles=5000, continuous_actions=False):

        materials_map = {(-0.5, -0.5): [0.5, 0.5, 0.5],
                         (0.5, 0.5): [0.9, 0.9, 0.9]}

        energy_locs = [(0.3, 0.3), (0.1, 0.1), 
                        np.random.uniform(-1, 1, 2), 
                        np.random.uniform(-1, 1, 2), 
                        np.random.uniform(-1, 1, 2), 
                        np.random.uniform(-1, 1, 2)]

        self.agent_lifetime = 100
        eating_distance = 0.3
        scenario = PetriScenario()
        world = scenario.make_world(num_agents=3, materials_map=materials_map, energy_locs=energy_locs, eating_distance=eating_distance)
        super().__init__(scenario, world, max_cycles, continuous_actions)
        self.metadata['name'] = "petri_env"

    def reset_agent_state(self, to_keep, parents):
        self.world.agents = [self.world.agents[i] for i in to_keep]
        # Added the ability for agents to die.
        self.scenario.add_new_agents(self.world, parents)

        self.agents = [agent.name for agent in self.world.agents]
        self._agent_selector = agent_selector(self.agents)
        self.possible_agents = self.agents[:]
        self._index_map = {agent.name: idx for idx, agent in enumerate(self.world.agents)}
        self._cumulative_rewards = {name: 0. for name in self.agents}

        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0

        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(low=0, high=1, shape=(space_dim,))
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(obs_dim,), dtype=np.float32)
        self.current_actions = [None] * self.num_agents
        self.rewards = {name: 0. for name in self.agents}
        if len(self.agents) > 0:
            self.agent_selection = self._agent_selector.reset()

    def _execute_world_step(self):
        # set action for each agent
        to_keep = []
        parents = []
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])
            agent.step_alive += 1

            if agent.can_reproduce:
                agent.can_reproduce = False
                parents.append(agent)
            
            if agent.step_alive < self.agent_lifetime:
                to_keep.append(i)
            else:
                print("Killing someone!")
        self.world.step()
        global_reward = 0.
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = global_reward * (1 - self.local_ratio) + agent_reward * self.local_ratio
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

        self.reset_agent_state(to_keep, parents=parents)

    def observe(self, agent):
        return self.scenario.observation(self.world.agents[self._index_map[agent]], self.world)#.astype(np.float32)

    def render(self, mode='human'):
        if not self.world.entities:
            return None
        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        self.render_geoms = None
        self.render_geoms_xform = None
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent._mpe_utils import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color[:3], alpha=0.5)
                else:
                    geom.set_color(*entity.color[:3])
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

            self.viewer.text_lines = []
            idx = 0
            for agent in self.world.agents:
                if not agent.silent:
                    tline = rendering.TextLine(self.viewer.window, idx)
                    self.viewer.text_lines.append(tline)
                    idx += 1

        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for idx, other in enumerate(self.world.agents):
            if other.silent:
                continue
            if np.all(other.state.c == 0):
                word = '_'
            elif self.continuous_actions:
                word = '[' + ",".join([f"{comm:.2f}" for comm in other.state.c]) + "]"
            else:
                word = alphabet[np.argmax(other.state.c)]

            message = (other.name + ' sends ' + word + '   ')

            self.viewer.text_lines[idx].set_text(message)

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
