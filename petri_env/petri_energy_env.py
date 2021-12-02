import random
from pettingzoo.mpe._mpe_utils import rendering
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.utils.agent_selector import agent_selector
from petri_env.petri_energy_scenario import PetriEnergyScenario
from petri_env.petri_core import PetriAgent, PetriMaterial
from rendering import make_square, make_triangle
import numpy as np
from gym import spaces
from pettingzoo.utils import wrappers
from utils import *
from collections import defaultdict

def make_env(raw_env):
    def env(config, neat_config, **kwargs):
        env = raw_env(config, neat_config, **kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env
    return env

class raw_env(SimpleEnv):

    def __init__(self, config, neat_config, continuous_actions=False):
        self.config = config
        self.init_num_agents = config["max_n_agents"]
        self.step_reward = config['step_reward']
        self.repr_reward = config['repr_reward']
        self.waste_lifetime = config['waste_lifetime']
        scenario = PetriEnergyScenario(config, neat_config)

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
        self.num_reproductions = 0

    def seed(self, val):
        pass

    def update_world_state(self):

        if not self.config['eat_action']:
            self.scenario.consume_resources(self.world)

        res_to_keep = [True for _ in range(len(self.world.landmarks))]
        ag_to_keep = [True for _ in range(len(self.world.agents))]

        res_eating_map = defaultdict(list)
        agent_eating_map = defaultdict(list)
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
                agent_eating_map[a].append(agent)

        for idx, landmark in enumerate(self.world.landmarks):
            if landmark.is_waste:
                landmark.waste_alive_time += 1

                if landmark.waste_alive_time > self.waste_lifetime:
                    res_to_keep[idx] = False

        ag_to_keep = self.resolve_collisions(res_eating_map, agent_eating_map, ag_to_keep)

        # Update resources
        self.world.landmarks = [self.world.landmarks[i] for i, to_keep in enumerate(res_to_keep) if to_keep]
        self.scenario.resource_generator.update_resources()
        
        # Update agents
        best_5 = None
        if self.env_step % 10 == 0:
            best_5 = self.update_reproducible_agents(self.world.agents)

        updated_agents = []
        for i, to_keep in enumerate(ag_to_keep):
            ag = self.world.agents[i]
            ag.reward += self.step_reward
            if to_keep and ag.energy_store > 0:
                updated_agents.append(ag)
            else:
                ag.tree_node = None
                del ag
                
        self.world.agents = updated_agents + self.agents_to_add
        self.agents_to_add = []
        self.env_step += 1
        # Added the ability for agents to die.
        if len(self.world.agents) < self.init_num_agents:
            # If ran out of agents, add new one
            if len(self.reproducible_agents) == 0:
                repr_agent = None
            else:
                repr_agent = random.sample(self.reproducible_agents, 1)[0]
            self.scenario.add_random_agent(self.world, self.env_step, repr_agent=repr_agent)


        self.world.calculate_distances()
        self.reset_maps()
        print("Num agents: {} {} {} {} {}".format(len(self.world.agents), self.env_step, len(self.reproducible_agents), self.num_reproductions, best_5))


    def resolve_collisions(self, res_eating_map, agent_eating_map, ag_to_keep):
        # Resolve resource eating collisions
        for res, agents in res_eating_map.items():
            dists = np.array([np.sum(np.square(res.state.p_pos - agent.state.p_pos)) for agent in agents])
            ag_idx = np.argmin(dists)
            agent = agents[ag_idx]
            agent.eat(res)
        
        # Resolve agent attacking collisions
        colliding_pairs = []
        for ag, agents in agent_eating_map.items():
            dists = np.array([np.sum(np.square(ag.state.p_pos - agent.state.p_pos)) for agent in agents])
            ag_idx = np.argmin(dists)
            agent = agents[ag_idx]
            if agent in agent_eating_map and ag in agent_eating_map[agent]:
                # Attacking each other at the same time
                colliding_pairs.append((ag, agent))
            else:
                agent.attack_agent(ag)
                idx = self.world.agents.index(ag)
                ag_to_keep[idx] = False

        for pair in colliding_pairs:
            ag1, ag2 = pair
            if (not ag1.dead) and (not ag2.dead):
                if ag1.energy_store > ag2.energy_store:
                    attacker, lost = ag1, ag2 
                else:
                    attacker, lost = ag2, ag1
                attacker.attack_agent(lost) 
                idx = self.world.agents.index(lost)
                lost.dead = True
                ag_to_keep[idx] = False

        return ag_to_keep

    def update_reproducible_agents(self, updated_agents):
        # Select top 50 agents
        for a in updated_agents:
            if a not in self.reproducible_agents:
                self.reproducible_agents.append(a)
        
        self.reproducible_agents = list(self.reproducible_agents)
        all_rewards = np.array([a.reward for a in self.reproducible_agents])
        best_ag_indices = np.argsort(all_rewards)[-10:]
        self.reproducible_agents = [self.reproducible_agents[i] for i in best_ag_indices]
        best_5 = [ag.reward for ag in self.reproducible_agents[-5:]]
        self.reproducible_agents = self.reproducible_agents

        return best_5

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
                    a = self.scenario.reproduce_agent(agent, world, self.env_step)
                    if a is not None:
                        self.num_reproductions += 1
                        self.agents_to_add.append(a)

                        # Store the agent if it reproduced properly
                        agent.reward += self.repr_reward
                act_id = 0
                if self.config["produce_res_action"]:
                    act_id += 1
                    if interact_act == act_id:
                        # Produce resource
                        self.scenario.produce_resource(agent, world)
                if self.config["eat_action"]:
                    act_id += 1
                    if interact_act == act_id:
                        # Eat resource
                        self.scenario.eat_resource(agent, world)                            
                if self.config["attack_action"]:
                    act_id += 1
                    if interact_act == act_id:
                        # Eat resource
                        self.scenario.attack_agent(agent, world)                            

            sensitivity = 2.0
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
            self.viewer.set_max_size(self.config["world_bound"] + 1)


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
                if isinstance(entity, PetriMaterial):
                    geom = make_square(entity.size  * 1)
                    geom.set_color(*entity.color[:3])
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    self.render_geoms.append(geom)
                    self.render_geoms_xform.append(xform)
                else:
                    geom_cons = make_triangle(entity.size  * 1.2)
                    geom_cons.set_color(*entity.consumes[:3], alpha=1)

                    xform = rendering.Transform()
                    geom_cons.add_attr(xform)
                    self.render_geoms.append(geom_cons)
                    self.render_geoms_xform.append(xform)

                    geom_color = rendering.make_circle(entity.size * 1.2)
                    geom_color.set_color(*entity.color[:3], alpha=1)

                    xform = rendering.Transform()
                    geom_color.add_attr(xform)
                    self.render_geoms.append(geom_color)
                    self.render_geoms_xform.append(xform)


                    geom_prod = make_square(entity.size * 1.2)
                    geom_prod.set_color(*entity.produces[:3], alpha=1)

                    xform = rendering.Transform()
                    geom_prod.add_attr(xform)
                    self.render_geoms.append(geom_prod)
                    self.render_geoms_xform.append(xform)


            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)


        # update geometry positions
        idx = 0
        for e, entity in enumerate(active_entities):
            if isinstance(entity, PetriAgent):
                self.render_geoms_xform[idx].set_translation(*entity.state.p_pos + np.array([0, 0.15]))
                self.render_geoms_xform[idx + 1].set_translation(*entity.state.p_pos)
                self.render_geoms_xform[idx + 2].set_translation(*entity.state.p_pos - np.array([0, 0.15]))
                idx += 3
            else:
                self.render_geoms_xform[idx].set_translation(*entity.state.p_pos)
                idx += 1
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


env = make_env(raw_env)
