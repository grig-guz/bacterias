import numpy as np

from petri_env.petri_core import PetriAgent, PetriEnergy, PetriMaterial, PetriWorld
from pettingzoo.mpe._mpe_utils.core import Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from policies.simple_policy import *
import copy
from utils import *


class PetriScenario(BaseScenario):

    def __init__(self, agents_produce_resources=False):
        super().__init__()
        self.visibility = 3
        self.agents_produce_resouces = agents_produce_resources

    def make_world(self, num_agents, materials_map, energy_locs, eating_distance):
        world = PetriWorld(eating_distance)
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
        # add landmarks
        world.landmarks = [PetriMaterial(loc, color) for loc, color in materials_map.items()] \
                            + [PetriEnergy(loc) for loc in energy_locs]

        for i, landmark in enumerate(world.landmarks):
            landmark.name = '%s %d'.format(landmark.__class__.__name__, i)
            landmark.collide = False
            landmark.movable = False
        
        return world            

    def reset_world(self, world, np_random):

        # random properties for agents
        #for i, agent in enumerate(world.agents):
        #    agent.color = np.array([1, 0, 0])
        # random properties for landmarks
        #for i, landmark in enumerate(world.landmarks):
        #    landmark.color = np.array([0.75, 0.75, 0.75])
        #world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for _, landmark in enumerate(world.landmarks):
            #print("Before", landmark.state.p_pos)
            #landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            #print("After", landmark.state.p_pos)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        return 0

    """
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
    """
    def observation(self, agent, world):
        # GCN observation
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        landmark_positions = world.resource_positions
        agent_positions = world.agent_positions


        agents_states = []
        if len(agent_positions) > 0:
            agents_states = self.get_features(agent, world.agents, agent_positions, self.get_agent_features, world)

        landmark_states = []
        if len(landmark_positions) > 0:
            landmark_states = self.get_features(agent, world.landmarks, landmark_positions, self.get_landmark_features, world)
            
        return [np.array(agents_states), 
                np.array(landmark_states), 
                np.concatenate([agent.state.p_pos, agent.state.p_vel, agent.color, agent.consumes, agent.produces])]

    """
    def observation(self, agent, world):
        # CNN observation
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        landmark_positions = world.resource_positions
        agent_positions = world.agent_positions


        to_remain, min_idx = dist_util(, self.visibility)

        return np.concatenate([agent.state.p_vel] + entity_pos)
    """

    def consume_resources(self, world):
        a_pos = np.array(world.agent_positions)
        r_pos = np.array(world.resource_positions)

        if len(r_pos) == 0 or len(a_pos) == 0:
            return

        to_remain, min_dists_idx = dist_util(r_pos, a_pos, world.eating_distace)

        if (to_remain == False).any():
            print("Removing!")
        
        for i, val in enumerate(to_remain):
            if not val:
                # If resource is eaten, the landmark is inactive
                # and the agent can reproduce.
                curr_agent = world.agents[min_dists_idx[i]]
                curr_agent.can_reproduce = True

                if self.agents_produce_resouces:
                    if isinstance(world.landmarks[i], PetriEnergy):
                        new_loc = np.random.uniform(-5, 5, 2)
                        world.landmarks[i] = PetriEnergy(new_loc)
                    else:
                        new_loc = curr_agent.state.p_loc + np.random.uniform(-1, 1, 2)
                        world.landmarks[i] = PetriMaterial(new_loc, curr_agent.produces)
                else:
                    world.landmarks[i].is_active = False


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

    def get_features(self, agent, entity_list, entity_dist_list, feat_func, world):
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
