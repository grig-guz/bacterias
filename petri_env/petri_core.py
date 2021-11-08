import numpy as np
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from utils import *

class PetriEnergy(Landmark):

    def __init__(self, loc):
        super().__init__()
        self.color = np.array([1, 1, 0])
        self.resource_type = "energy"
        self.state.p_pos = loc


class PetriMaterial(Landmark):

    def __init__(self, loc, color):
        super().__init__()
        self.resource_type = "material"
        self.state.p_pos = np.array(loc)
        self.color = np.array(color)


class PetriAgent(Agent):

    def __init__(self, loc, consumes, produces, material, policy=None):
        super().__init__()

        # TODO: Fix this
        self.state.p_pos = np.array(loc)
        self.consumes = np.array(consumes)
        self.produces = np.array(produces)
        self.color = np.array(material)
        self.can_reproduce = False
        self.step_alive = 0
        self.policy = policy


class PetriWorld(World):

    def __init__(self, eating_distance):
        super().__init__()
        self.eating_distace = eating_distance

    @property
    def agent_positions(self):
        return [agent.state.p_pos for agent in self.agents]

    @property
    def resource_positions(self):
        return [resource.state.p_pos for resource in self.landmarks]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

        self.consume_resources()

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt
            if entity.state.p_pos[0] > 5:
                entity.state.p_pos[0] = -5
            elif entity.state.p_pos[0] < -5:
                entity.state.p_pos[0] = 5

            if entity.state.p_pos[1] > 5:
                entity.state.p_pos[1] = -5
            elif entity.state.p_pos[1] < -5:
                entity.state.p_pos[1] = 5


    def consume_resources(self):
        a_pos = np.array(self.agent_positions)
        r_pos = np.array(self.resource_positions)

        if len(r_pos) == 0 or len(a_pos) == 0:
            return

        to_remain, min_dists_idx = dist_util(r_pos, a_pos, self.eating_distace)

        if (to_remain == False).any():
            print("Removing!")
        
        self.landmarks = [self.landmarks[i] for i, val in enumerate(to_remain) if val]

        # Agents that ate something can reproduce.
        for i, val in enumerate(to_remain):
            if val == False:
                self.agents[min_dists_idx[i]].can_reproduce = True 
        