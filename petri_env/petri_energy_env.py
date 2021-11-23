import random
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils import rendering
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.utils.agent_selector import agent_selector
from petri_env.petri_energy_scenario import PetriEnergyScenario
from petri_env.petri_core import PetriEnergy, PetriMaterial
from rendering import make_square
import numpy as np
from gym import spaces
from pettingzoo.utils import wrappers
from utils import *
from random import sample
import copy
from collections import defaultdict

def make_env(raw_env):
    def env(config, **kwargs):
        env = raw_env(config, **kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env
    return env

class raw_env(SimpleEnv):

    def __init__(self, config, continuous_actions=False):
        self.config = config
        self.use_energy_resource = config['use_energy_resource']
        self.init_num_agents = config["max_n_agents"]
        scenario = PetriEnergyScenario(config)

        materials_map = {(-0.5, -0.5): [0.5, 0.5, 0.5],
                         (0.5, 0.5): [0.9, 0.9, 0.9]}

        energy_locs = [(0.3, 0.3), (0.1, 0.1), 
                        np.random.uniform(-1, 1, 2), 
                        np.random.uniform(-1, 1, 2), 
                        np.random.uniform(-1, 1, 2), 
                        np.random.uniform(-1, 1, 2)]

        world = scenario.make_world(materials_map=materials_map, energy_locs=energy_locs)
        self.reproducible_agents = []
        super().__init__(scenario, world, config['max_cycles'], continuous_actions)
        self.metadata['name'] = "petri_env"
        self.env_step = 0

    def update_world_state(self):

        if not self.config['eat_action']:
            self.scenario.consume_resources(self.world)

        res_to_keep = [True for _ in range(len(self.world.landmarks))]
        ag_to_keep = [True for _ in range(len(self.world.agents))]

        res_eating_map = defaultdict(list)
        for agent in self.world.agents:
            if agent.currently_eating is not None:
                landmark = agent.currently_eating
                agent.currently_eating = None
                idx = self.world.landmarks.index(landmark)
                landmark.is_active = False
                if landmark.is_waste:
                    res_to_keep[idx] = False
                res_eating_map[landmark].append(agent)
            elif agent.currently_attacking is not None:
                a = agent.currently_attacking
                agent.currently_attacking = None
                idx = self.world.agents.index(a)
                ag_to_keep[idx] = False
                agent.attack_agent(a)

        self.resolve_collisions(res_eating_map)

        self.world.landmarks = [self.world.landmarks[i] for i, to_keep in enumerate(res_to_keep) if to_keep]
        self.scenario.resource_generator.update_resources()
        self.world.agents = [self.world.agents[i] for i, to_keep in enumerate(ag_to_keep) if to_keep and self.world.agents[i].energy_store > 0] + self.agents_to_add
        print("Num agents: {} {} {}".format(len(self.world.agents), self.env_step, len(self.reproducible_agents)))
        self.agents_to_add = []
        self.env_step += 1
        # Added the ability for agents to die.
        if self.env_step % 5 == 0 and len(self.agents) < self.init_num_agents:
            # If ran out of agents, add new one
            if self.reproducible_agents == []:
                repr_agent = None
            else:
                repr_agent = random.sample(self.reproducible_agents, 1)[0]
            self.scenario.add_random_agent(self.world, repr_agent=repr_agent)

        self.world.calculate_distances()
        self.reset_maps()

    def resolve_collisions(self, res_eating_map):
        # Resolve resource eating collisions
        for res, agents in res_eating_map.items():
            agent = random.sample(agents, 1)[0]
            agent.eat(res)
        # Resolve agent attacking collisions


    def _execute_world_step(self):
        # set action for each agent
        self.agents_to_add = []
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            scenario_action.append(action)
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.world)
            agent.step_alive += 1
        self.world.step()
        for agent in self.world.agents:
            self.rewards[agent.name] = 0

        self.update_world_state()

    def _set_action(self, action, agent, world, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        if agent.movable:
            # physical action
            move_act, interact_act = action[0]
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            else:
                # process discrete action
                #if move_act == 0:
                    #print("idle", agent.energy_store)
                #    agent.idle()
                if move_act == 0:
                    agent.action.u[0] = -1.0
                    agent.move()
                if move_act == 1:
                    agent.action.u[0] = +1.0
                    agent.move()
                if move_act == 2:
                    agent.action.u[1] = -1.0
                    agent.move()
                if move_act == 3:
                    agent.action.u[1] = +1.0
                    agent.move()
                
                # Agency variations:
                if interact_act == 0:
                    # Reproduce
                    agent.idle()
                    a = self.scenario.reproduce_agent(agent, world)
                    if a is not None:
                        self.agents_to_add.append(a)
                        saved_ag = copy.deepcopy(agent)
                        saved_ag.lineage_length = 1
                        self.reproducible_agents.append(saved_ag)
                        self.reproducible_agents = self.reproducible_agents[-50:]
                act_id = 0
                if self.config["produce_res_action"]:
                    act_id += 1
                    if interact_act == act_id:
                        # Produce resource
                        agent.idle()
                        self.scenario.produce_resource(agent, world)
                if self.config["eat_action"]:
                    act_id += 1
                    if interact_act == act_id:
                        # Eat resource
                        agent.idle()
                        self.scenario.eat_resource(agent, world)                            
                if self.config["attack_action"]:
                    act_id += 1
                    if interact_act == act_id:
                        # Eat resource
                        self.scenario.attack_agent(agent, world)                            

            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def observe(self, agent):
        return self.scenario.observation(self.world.agents[self._index_map[agent]], self.world)#.astype(np.float32)

    def reset_maps(self):
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
            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(low=0, high=1, shape=(9,))
            else:
                self.action_spaces[agent.name] = spaces.Discrete(9)
            self.observation_spaces[agent.name] = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(obs_dim,), dtype=np.float32)

        self.current_actions = [None] * self.num_agents
        self.rewards = {name: 0. for name in self.agents}
        if len(self.agents) > 0:
            self.agent_selection = self._agent_selector.reset()


    def render(self, mode='human'):
        if not self.world.entities:
            return None
        if self.viewer is None:
            self.viewer = rendering.Viewer(900, 900)
            self.viewer.set_max_size(5)


        # create rendering geometry
        self.render_geoms = None
        self.render_geoms_xform = None
        active_entities = [ent for ent in self.world.entities if ent.is_active]
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent._mpe_utils import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in active_entities:
                if isinstance(entity, PetriEnergy):
                    geom = rendering.make_circle(entity.size)
                elif isinstance(entity, PetriMaterial):
                    geom = make_square(entity.size)
                else:
                    geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.consumes[:3], alpha=0.5)
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
        all_poses = [entity.state.p_pos for entity in active_entities]
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(active_entities):
            if entity.is_active:
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
