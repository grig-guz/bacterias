import numpy as np

from petri_env.petri_core import PetriAgent, PetriEnergy, PetriMaterial, PetriWorld
from pettingzoo.mpe._mpe_utils.core import Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario



class PetriScenario(BaseScenario):

    def __init__(self):
        super().__init__()

    def make_world(self, num_agents, materials_map, energy_locs, eating_distance):
        world = PetriWorld(eating_distance)
        # add agents

        world.agents = []
        for i in range(num_agents):
            loc = np.random.uniform(-1, 1, 2)
            color = np.array([1, 0, 0])
            consumes = np.random.uniform(-1, 1, 3)
            produces = np.random.uniform(-1, 1, 3)
            agent = PetriAgent(loc=loc, consumes=consumes, produces=produces, material=color)
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

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)

    def add_new_agents(self, world, new_a_locs):
        for loc in new_a_locs:
            loc += np.random.uniform(-0.1, 0.1, 2)
            color = np.array([1, 0, 0])
            consumes = np.random.uniform(-1, 1, 3)
            produces = np.random.uniform(-1, 1, 3)
            agent = PetriAgent(loc=loc, consumes=consumes, produces=produces, material=color)
            agent.name = f'agent_{world.agent_counter}'
            world.agent_counter += 1
            agent.collide = False
            agent.silent = True
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            world.agents.append(agent)
