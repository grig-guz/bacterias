import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World

class PetriEnergy(Landmark):

    def __init__(self, loc):
        super().__init__()
        self.color = np.array([1, 1, 0])
        self.state.p_pos = loc


class PetriMaterial(Landmark):

    def __init__(self, loc, color):
        super().__init__()
        self.state.p_pos = np.array(loc)
        self.color = np.array(color)

class PetriAgent(Agent):

    def __init__(self, loc, consumes, produces, material, policy):
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

    def consume_resources(self):
        a_pos = np.array(self.agent_positions)
        r_pos = np.array(self.resource_positions)

        if len(r_pos) == 0 or len(a_pos) == 0:
            return

        r_a_dists = euclidean_distances(r_pos, a_pos)
        min_dists_idx = np.argmin(r_a_dists, axis=1)

        to_remain = r_a_dists[np.arange(len(r_pos)), min_dists_idx] > self.eating_distace

        if (to_remain == False).any():
            print("Removing!")
        
        self.landmarks = [self.landmarks[i] for i, val in enumerate(to_remain) if val]

        # Agents that ate something can reproduce.
        for i, val in enumerate(to_remain):
            if val == False:
                self.agents[min_dists_idx[i]].can_reproduce = True 