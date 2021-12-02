import numpy as np
import copy
import os
np.set_printoptions(threshold=np.inf)

from pettingzoo.mpe._mpe_utils.scenario import BaseScenario

from petri_env.petri_core import PetriEnergyAgent, PetriNeatAgent, PetriMaterial, PetriWorld
from petri_env.resource_generator import BimodalResourceGenerator, RandomResourceGenerator, FixedResourceGenerator
from policies.simple_policy import *

from utils import *

RANDOM_REC_GEN = "random"
FIXED_REC_GEN = "fixed"
BIMODAL_REC_GEN = "bimodal"


class PetriEnergyScenario(BaseScenario):

    def __init__(self, config, neat_config):
        super().__init__()
        self.config = config
        self.neat_config = neat_config
        self.sigma = config['sigma']
        self.model_sigma = config['model_sigma']
        self.eating_distance = config['eating_distance']
        self.res_gen_type = config['res_gen_type']
        self.world_bound = config['world_bound']
        self.num_resources = config["num_resources"]
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
        self.select_by_distance = config["select_by_distance"]



    def make_world(self, materials_map, energy_locs):
        world = PetriWorld(self.config)
        # add agents
        world.agents = []
        world.agent_counter = 0
        self.add_random_agent(world, 0)

        if self.res_gen_type == RANDOM_REC_GEN:
            self.resource_generator = RandomResourceGenerator(self.config, world=world)
            self.resource_generator.generate_initial_resources()
        elif self.res_gen_type == BIMODAL_REC_GEN:
            self.resource_generator = BimodalResourceGenerator(self.config, world=world)
            self.resource_generator.generate_initial_resources()
        else:
            print("Unknown resource generator type")
            raise Exception

        world.calculate_distances()
        return world            

    def add_random_agent(self, world, timestep, repr_agent=None):
        loc = np.random.uniform(-self.world_bound, self.world_bound, 2)
        color = np.random.uniform(0, 1, 3)
        consumes = np.random.uniform(0, 1, 3)
        produces = np.random.uniform(0, 1, 3)

        if repr_agent is None:
            consumes = np.array([1, 0, 0])
            policy = GCNAttnPolicy(obs_dim=8, action_dim=self.action_dim, sigma=self.model_sigma)
            agent = PetriEnergyAgent(self.config, loc=loc, consumes=consumes, produces=produces, material=color, policy=policy)
        else:
            agent = copy.deepcopy(repr_agent)
            agent.state.p_pos = loc
            agent.mutate()
        agent.tree_node.timestep = timestep
        agent.state.p_vel = np.zeros(2)
        agent.name = f'agent_{world.agent_counter}'
        agent.collide = False
        agent.silent = True
        world.agents.append(agent)
        world.agent_counter += 1


    def reset_world(self, world, np_random):
        pass

    def reward(self, agent, world):
        return 0
        
    def observation(self, agent, world):
        # GCN observation
        # get positions of all entities in this agent's reference frame
        landmarks = world.active_resources
        agents = world.agents
        agent_id = world.agents.index(agent)

        landmark_states = []
        if len(landmarks) > 0:
            landmark_states = self.get_features(agent, agent_id, landmarks, world.agent_res_distances, self.get_landmark_features)
        
        agents_states = []
        if len(agents) > 0:
            agents_states = self.get_features(agent, agent_id, agents, world.agent_agent_distances, self.get_agent_features)
            while len(agents_states) < self.select_by_distance:
                agents_states.append(-2 * np.ones(13))
            agents_states = np.stack(agents_states)
        c_agent_obs = np.concatenate([agent.state.p_pos / self.world_bound, 
                            agent.state.p_vel, 
                            np.array([agent.energy_store]) / self.max_energy, 
                            agent.color, 
                            agent.consumes, 
                            agent.produces])
        return [np.array(agents_states), 
                np.array(landmark_states), 
                c_agent_obs]
    

    def get_features(self, agent, agent_id, entity_list, distances, feat_func):

        acc = []
        if self.select_by_distance > 0:
            indices = np.argsort(distances[agent_id])[:self.select_by_distance]
            for i in indices:
                if agent != entity_list[i]:
                    acc.append(feat_func(entity_list[i], agent))
        return acc

    def get_agent_features(self, agent1, agent2):
        dist_diff = (agent1.state.p_pos - agent2.state.p_pos)
        vel = agent1.state.p_vel
        color = agent1.color
        consumes = agent1.consumes
        produces = agent1.produces
        return np.concatenate([dist_diff, vel, color, consumes, produces])

    def get_landmark_features(self, landmark, agent):
        dist_diff = (landmark.state.p_pos -  agent.state.p_pos)
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
            print("$$$$$ produced resouce")

    def eat_resource(self, agent, world):
        if len(world.landmarks) > 0:
            agent_id = world.agents.index(agent)
            src_dst_dists = world.agent_res_distances[agent_id]
            closest_idx = np.argmin(src_dst_dists)
            closest_dist = src_dst_dists[closest_idx]
            if closest_dist < self.eating_distance:
                agent.assign_eat(closest_idx, world)

    def reproduce_agent(self, agent, world, timestep):
        if agent.reproduce():
            print("***** Reproduced, lineage length:", agent.lineage_length + 1)
            new_agent = copy.deepcopy(agent)
            new_agent.mutate()
            new_agent.step_alive = 0
            new_agent.name = f'agent_{world.agent_counter}'
            world.agent_counter += 1
            new_agent.state.p_vel = np.zeros(world.dim_p)
            new_agent.state.c = np.zeros(world.dim_c)
            new_agent.lineage_length += 1

            new_agent.tree_node.parent = agent.tree_node
            agent.tree_node.children.append(new_agent.tree_node)
            
            new_agent.tree_node.consumes = copy.deepcopy(new_agent.consumes)
            new_agent.tree_node.produces = copy.deepcopy(new_agent.produces)
            new_agent.tree_node.color = copy.deepcopy(new_agent.color)
            new_agent.tree_node.timestep = timestep
            return new_agent
        return None

    def attack_agent(self, agent, world):
        if len(world.agents) > 1:
            agent_id = world.agents.index(agent)
            src_dst_dists = world.agent_agent_distances[agent_id]
            closest_idx = np.argmin(src_dst_dists)
            closest_dist = src_dst_dists[closest_idx]
            closest_agent = world.agents[closest_idx]
            if closest_dist < self.eating_distance and agent.can_attack(closest_agent):
                print(">>>>> Attacked an agent!")
                agent.assign_attack(closest_agent)
