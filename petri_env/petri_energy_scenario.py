import numpy as np
import copy

from pettingzoo.mpe._mpe_utils.scenario import BaseScenario

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
        self.agents_produce_resouces = config['agents_produce_resources']
        self.world_bound = config['world_bound']
        self.num_resources = config["num_resources"]
        self.num_agents = config['num_agents']
        self.recov_time = config['recov_time']
        self.use_energy_resource = config['use_energy_resource']
        self.action_dim = 7
        if config["attack_action"]:
            self.action_dim += 1
        if config["eat_action"]:
            self.action_dim += 1



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

        return world            

    def add_random_agent(self, world):
        for _ in range(1):
            loc = np.random.uniform(-1, 1, 2)
            color = np.array([1., 0., 0.])
            consumes = np.random.uniform(-1, 1, 3)
            produces = np.random.uniform(-1, 1, 3)
            policy = GCNPolicy(obs_dim=8, action_dim=4, sigma=self.model_sigma)
            if self.use_energy_resource:
                agent = PetriAgent(loc=loc, consumes=consumes, produces=produces, material=color, policy=policy)
                agent.state.p_vel = np.zeros(2)
            else:
                agent = PetriEnergyAgent(self.config, loc=loc, consumes=consumes, produces=produces, material=color, policy=policy)
                agent.state.p_vel = np.zeros(2)
            agent.name = f'agent_{world.agent_counter}'
            agent.collide = False
            agent.silent = True
            world.agents.append(agent)
            world.agent_counter += 1


    def reset_world(self, world, np_random):

        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for _, landmark in enumerate(world.landmarks):
            landmark.state.p_vel = np.zeros(world.dim_p)


    def reward(self, agent, world):
        return 0


    def observation(self, agent, world):
        # GCN observation
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for landmark in world.landmarks:
            entity_pos.append(landmark.state.p_pos - agent.state.p_pos)

        landmark_positions = world.active_resource_positions
        agent_positions = world.agent_positions

        agents_states = []
        if len(agent_positions) > 0:
            agents_states = self.get_features(agent, world.agents, agent_positions, self.get_agent_features)

        landmark_states = []
        if len(landmark_positions) > 0:
            landmark_states = self.get_features(agent, world.active_resources, landmark_positions, self.get_landmark_features)
            
        return [np.array(agents_states), 
                np.array(landmark_states), 
                np.concatenate([agent.state.p_pos, agent.state.p_vel, agent.color, agent.consumes, agent.produces])]


    def consume_resources(self, world):
        a_pos = np.array(world.agent_positions)
        r_pos = np.array(world.resource_positions)

        if len(r_pos) == 0 or len(a_pos) == 0:
            return

        to_remain, min_dists_idx = dist_util(r_pos, a_pos, world.eating_distace)

        if (to_remain == False).any():
            print("Removing!")
        
        for i, val in enumerate(to_remain):
            if not val and world.landmarks[i].is_active:
                # If resource is eaten, the landmark is inactive
                # and the agent can reproduce.
                curr_agent = world.agents[min_dists_idx[i]]

                if self.agents_produce_resouces:
                    if isinstance(world.landmarks[i], PetriEnergy):
                        new_loc = np.random.uniform(-self.world_bound, self.world_bound, 2)
                        world.landmarks[i] = PetriEnergy(new_loc)
                        curr_agent.consumed_energy = True
                    else:
                        new_loc = curr_agent.state.p_pos + np.random.uniform(-0.1, 0.1, 2)
                        world.landmarks[i] = PetriMaterial(new_loc, curr_agent.produces)
                        curr_agent.consumed_material = True

                if (not self.use_energy_resource and curr_agent.consumed_material) or (curr_agent.consumed_material and curr_agent.consumed_energy):
                    curr_agent.can_reproduce = True
                    curr_agent.consumed_material = False 
                    curr_agent.consumed_energy = False

                world.landmarks[i].is_active = False

        # TODO: Maybe make this be at the beginning of the function.
        self.resource_generator.update_resources()


    def get_features(self, agent, entity_list, entity_dist_list, feat_func):
        acc = []
        selected_list, min_idx = dist_util(np.array([agent.state.p_pos]), entity_dist_list, self.visibility)
        selected_list, min_idx = selected_list[0], min_idx[0]
        for i, selected in enumerate(selected_list):
            if selected:
                acc.append(feat_func(entity_list[min_idx[i]], agent))
        return acc


    def get_agent_features(self, agent1, agent2):
        dist_diff = agent1.state.p_pos - agent2.state.p_pos
        vel = agent1.state.p_vel
        color = agent1.color
        consumes = agent1.consumes
        produces = agent1.produces
        return np.concatenate([dist_diff, vel, color, consumes, produces])


    def get_landmark_features(self, landmark, agent):
        dist_diff = landmark.state.p_pos - agent.state.p_pos
        rtype = np.array([int(landmark.resource_type == "energy")])
        color = landmark.color
        return np.concatenate([dist_diff, rtype, color])

    def produce_resource(self, agent, world):
        if agent.can_produce_resource():
            new_loc = agent.state.p_pos + np.random.uniform(-0.1, 0.1, 2)
            res_color = agent.produces
            resource = PetriMaterial(new_loc, res_color)
            world.landmarks.append(resource)

    def eat_resource(self, agent, world):
        a_pos = np.array([agent.state.p_pos])
        r_pos = np.array(world.resource_positions)
        if len(r_pos) > 0:
            src_dst_dists = euclidean_distances(a_pos, r_pos)[0]
            closest_idx = np.argmin(src_dst_dists)
            closest_dist = src_dst_dists[closest_idx]
            if closest_dist < self.eating_distance:
                print("SUCCESS RESOURCE!")
                agent.assign_eat(closest_idx, world)

    def reproduce_agent(self, agent, world):
        if agent.reproduce():
            new_agent = copy.deepcopy(agent)
            new_agent.mutate()
            new_agent.step_alive = 0
            new_agent.name = f'agent_{world.agent_counter}'
            world.agent_counter += 1
            new_agent.state.p_vel = np.zeros(world.dim_p)
            new_agent.state.c = np.zeros(world.dim_c)
            return new_agent
        return None

    def attack_agent(self, agent, world):
        a_pos = np.array([agent.state.p_pos])
        other_agents = [a for a in world.agents if a != agent]
        if len(other_agents) == 0:
            return
        r_pos = np.array([a.state.p_pos for a in other_agents])
        if len(r_pos) > 0:
            src_dst_dists = euclidean_distances(a_pos, r_pos)[0]
            closest_idx = np.argmin(src_dst_dists)
            closest_dist = src_dst_dists[closest_idx]
            if closest_dist < self.eating_distance:
                print("SUCCESS ATTACK!")
                agent.assign_attack(other_agents[closest_idx])
