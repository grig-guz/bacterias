import numpy as np
import copy

from pettingzoo.mpe._mpe_utils.scenario import BaseScenario

from petri_env.petri_core import PetriAgent, PetriEnergy, PetriMaterial, PetriWorld
from petri_env.resource_generator import RandomResourceGenerator, FixedResourceGenerator
from policies.simple_policy import *

from utils import *

RANDOM_REC_GEN = "random"
FIXED_REC_GEN = "fixed"


class PetriScenario(BaseScenario):


    def __init__(self, res_gen_type, world_bound=7, agents_produce_resources=True):
        super().__init__()
        self.visibility = 3
        self.res_gen_type = res_gen_type
        self.agents_produce_resouces = agents_produce_resources
        self.world_bound = world_bound


    def make_world(self, num_agents, materials_map, energy_locs, eating_distance):
        world = PetriWorld(self.world_bound, eating_distance)
        # add agents

        world.agents = []
        for i in range(num_agents):
            loc = np.random.uniform(-1, 1, 2)
            color = np.array([1., 0., 0.])
            consumes = np.random.uniform(-1, 1, 3)
            produces = np.random.uniform(-1, 1, 3)
            policy = GCNPolicy(obs_dim=8, action_dim=4, sigma=0.1)
            agent = PetriAgent(loc=loc, consumes=consumes, produces=produces, material=color, policy=policy)
            agent.name = f'agent_{i}'
            agent.collide = False
            agent.silent = True
            world.agents.append(agent)
        world.agent_counter = num_agents

        recov_time = 30

        if self.res_gen_type == RANDOM_REC_GEN:
            self.resource_generator = RandomResourceGenerator(world=world, recov_time=recov_time, world_bound=self.world_bound, num_resources=70)
            self.resource_generator.generate_initial_resources()
        elif self.res_gen_type == FIXED_REC_GEN:
            self.resource_generator = FixedResourceGenerator(world=world, recov_time=recov_time)
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
                curr_agent.can_reproduce = True

                if self.agents_produce_resouces:
                    if isinstance(world.landmarks[i], PetriEnergy):
                        new_loc = np.random.uniform(-self.world_bound, self.world_bound, 2)
                        world.landmarks[i] = PetriEnergy(new_loc)
                    else:
                        # TODO: Make this close to agent eventually
                        new_loc = np.random.uniform(-self.world_bound, self.world_bound, 2)
                        world.landmarks[i] = PetriMaterial(new_loc, curr_agent.produces)

                world.landmarks[i].is_active = False

        # TODO: Maybe make this be at the beginning of the function.
        self.resource_generator.update_resources()


    def add_new_agents(self, world, parents):
        for _, agent in enumerate(parents):
            new_agent = copy.deepcopy(agent)
            new_agent.mutate()
            new_agent.step_alive = 0
            new_agent.name = f'agent_{world.agent_counter}'
            world.agent_counter += 1
            new_agent.state.p_vel = np.zeros(world.dim_p)
            new_agent.state.c = np.zeros(world.dim_c)
            world.agents.append(new_agent)


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
