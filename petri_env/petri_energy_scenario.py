import numpy as np
import copy
import os
np.set_printoptions(threshold=np.inf)
import time

from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from torch._C import _resolve_type_from_object

from petri_env.petri_core import PetriAgent, PetriEnergyAgent, PetriEnergy, PetriMaterial, PetriWorld
from petri_env.resource_generator import RandomResourceGenerator, FixedResourceGenerator
from policies.simple_policy import *

from utils import *

RANDOM_REC_GEN = "random"
FIXED_REC_GEN = "fixed"


class PetriEnergyScenario(BaseScenario):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sigma = config['sigma']
        self.model_sigma = config['model_sigma']
        self.eating_distance = config['eating_distance']
        self.visibility = config['visibility']
        self.res_gen_type = config['res_gen_type']
        self.world_bound = config['world_bound']
        self.num_resources = config["num_resources"]
        self.num_agents = config['num_agents']
        self.recov_time = config['recov_time']
        self.use_energy_resource = config['use_energy_resource']
        self.obs_size = config['obs_size']
        self.action_dim = 1
        self.max_energy = config['max_energy']
        if config["attack_action"]:
            self.action_dim += 1
        if config["eat_action"]:
            self.action_dim += 1
        if config["produce_res_action"]:
            self.action_dim += 1
        self.novelty_buffer = []
        self.count = 0



    def make_world(self, materials_map, energy_locs):
        world = PetriWorld(self.config)
        # add agents
        world.agents = []
        world.agent_counter = 0
        for _ in range(self.num_agents):
            self.add_random_agent(world)

        if self.res_gen_type == RANDOM_REC_GEN:
            self.resource_generator = RandomResourceGenerator(self.config, world=world)
            self.resource_generator.generate_initial_resources()
        elif self.res_gen_type == FIXED_REC_GEN:
            self.resource_generator = FixedResourceGenerator(self.config, world=world)
            # add landmarks
            world.landmarks = [PetriMaterial(loc, color) for loc, color in materials_map.items()] \
                                + [PetriEnergy(loc) for loc in energy_locs]

            for i, landmark in enumerate(world.landmarks):
                landmark.name = '%s %d'.format(landmark.__class__.__name__, i)
                landmark.collide = False
                landmark.movable = False

        else:
            print("Unknown resource generator type")
            raise Exception

        world.calculate_distances()
        return world            

    def add_random_agent(self, world, repr_agent=None):
        for _ in range(1):
            loc = np.random.uniform(-self.world_bound, self.world_bound, 2)
            color = np.array([1., 0., 0.])
            #consumes = np.random.uniform(0, 1, 3)
            produces = np.random.uniform(0, 1, 3)
            consumes = np.array([1., 0., 0.])
            #produces = np.array([1., 0., 0.])
            if repr_agent is None:
                policy = GCNAttnPolicy(obs_dim=8, action_dim=self.action_dim, sigma=self.model_sigma)
            else:
                policy = copy.deepcopy(repr_agent.policy)
                policy.mutate()
            if self.use_energy_resource:
                agent = PetriAgent(loc=loc, consumes=consumes, produces=produces, material=color, policy=policy)
            else:
                agent = PetriEnergyAgent(self.config, loc=loc, consumes=consumes, produces=produces, material=color, policy=policy)

            agent.state.p_vel = np.zeros(2)
            agent.name = f'agent_{world.agent_counter}'
            agent.collide = False
            agent.silent = True
            world.agents.append(agent)
            world.agent_counter += 1


    def reset_world(self, world, np_random):
        pass
        """
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-self.world_bound, self.world_bound, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for _, landmark in enumerate(world.landmarks):
            landmark.state.p_vel = np.zeros(world.dim_p)
        """

    def reward(self, agent, world):
        return 0

    def observation(self, agent, world):
        # GCN observation
        # get positions of all entities in this agent's reference frame
        cell = copy.deepcopy(world.cell)
        x, y = agent.state.p_pos
        x = np.digitize(x, world.bins) - 1
        y = np.digitize(-y, world.bins) - 1
        # Agent identity
        cell[12, y, x] = 1
        # Agent energy amount
        cell[13, y, x] = agent.energy_store / self.max_energy

        cell = self.compress_cell(cell, y, x)
        """
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'test.npy')
        with open(filename, 'wb') as f:
            np.save(f, cell)
            print("SAVED")
        if self.count > 0:
            time.sleep(30)
        self.count += 1
        """
        return cell

    def compress_cell(self, cell, x, y):
        # 
        new_cell = -np.ones((cell.shape[0], 60 + self.obs_size * 2, 60 + self.obs_size * 2)) * 2
        new_cell[:, self.obs_size:60+self.obs_size, self.obs_size:60+self.obs_size] = cell
        x += 15
        y += 15
        return new_cell[:, y-self.obs_size:y+self.obs_size, x-self.obs_size:x+self.obs_size]


    
    def observation(self, agent, world):
        # GCN observation
        # get positions of all entities in this agent's reference frame
        landmarks = world.active_resources
        agents = world.agents
        agent_id = world.agents.index(agent)

        agents_states = []
        if len(agents) > 0:
            agents_states = self.get_features(agent, agent_id, agents, world.agent_agent_distances, self.get_agent_features)
        agents_states = []
        landmark_states = []
        if len(landmarks) > 0:
            landmark_states = self.get_features(agent, agent_id, landmarks, world.agent_res_distances, self.get_landmark_features)

        return [np.array(agents_states), 
                np.array(landmark_states), 
                np.concatenate([agent.state.p_pos / self.world_bound, 
                        agent.state.p_vel, 
                        #np.array([self.world_bound - agent.state.p_pos[0],
                        #            self.world_bound - agent.state.p_pos[1],
                        #            -self.world_bound - agent.state.p_pos[0],
                        #            -self.world_bound - agent.state.p_pos[1]]) / self.world_bound,
                        np.array([agent.energy_store]) / self.max_energy, 
                        agent.color, 
                        agent.consumes, 
                        agent.produces])]
    

    def get_features(self, agent, agent_id, entity_list, distances, feat_func):

        acc = []
        selected_list = distances[agent_id] < self.visibility
        for i, selected in enumerate(selected_list):
            if selected and agent != entity_list[i]:
                acc.append(feat_func(entity_list[i], agent))
        return acc

    def get_agent_features(self, agent1, agent2):
        dist_diff = (agent1.state.p_pos - agent2.state.p_pos)/ self.world_bound
        vel = agent1.state.p_vel
        color = agent1.color
        consumes = agent1.consumes
        produces = agent1.produces
        return np.concatenate([dist_diff, vel, color, consumes, produces])

    def get_landmark_features(self, landmark, agent):
        dist_diff = (landmark.state.p_pos -  agent.state.p_pos) / self.world_bound
        color = landmark.color
        return np.concatenate([dist_diff, color])

    def produce_resource(self, agent, world):
        if agent.can_produce_resource():
            new_loc = np.clip(agent.state.p_pos + np.random.uniform(-0.3, 0.3, 2), -self.world_bound, self.world_bound)
            res_color = agent.produces
            resource = PetriMaterial(new_loc, res_color)
            resource.is_waste = True
            agent.consumed_material = False
            world.landmarks.append(resource)
            print("produced resouce")

    def eat_resource(self, agent, world):
        if len(world.landmarks) > 0:
            agent_id = world.agents.index(agent)
            src_dst_dists = world.agent_res_distances[agent_id]
            closest_idx = np.argmin(src_dst_dists)
            closest_dist = src_dst_dists[closest_idx]
            if closest_dist < self.eating_distance:
                agent.assign_eat(closest_idx, world)

    def reproduce_agent(self, agent, world):
        if agent.reproduce():
            print("SUCCESS REPRODUCE, lineage length:", agent.lineage_length + 1)
            new_agent = copy.deepcopy(agent)
            new_agent.mutate()
            new_agent.step_alive = 0
            new_agent.name = f'agent_{world.agent_counter}'
            world.agent_counter += 1
            new_agent.state.p_vel = np.zeros(world.dim_p)
            new_agent.state.c = np.zeros(world.dim_c)
            new_agent.lineage_length += 1
            return new_agent
        return None

    def attack_agent(self, agent, world):
        if agent.attack():
            if len(world.agents) > 1:
                agent_id = world.agents.index(agent)
                src_dst_dists = world.agent_agent_distances[agent_id]
                closest_idx = np.argmin(src_dst_dists)
                closest_dist = src_dst_dists[closest_idx]
                if closest_dist < self.eating_distance:
                    print("SUCCESSFUL EATING!")
                    agent.assign_attack(other_agents[closest_idx])
