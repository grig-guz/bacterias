import numpy as np
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from utils import *


class PetriLandmark(Landmark):  # properties of landmark entities
    def __init__(self):
        super().__init__()
        self.is_active = True
        self.inactive_count = 0


class PetriEnergy(PetriLandmark):

    def __init__(self, loc):
        super().__init__()
        self.resource_type = "energy"
        self.state.p_pos = loc
        self.is_active = True
        self.color = np.array([1, 1, 0])


class PetriMaterial(PetriLandmark):

    def __init__(self, loc, color):
        super().__init__()
        self.resource_type = "material"
        self.state.p_pos = np.array(loc)
        self.color = np.array(color)


class PetriAgent(Agent):

    def __init__(self, loc, consumes, produces, material, policy=None):
        super().__init__()
        self.state.p_pos = np.array(loc)
        self.consumes = np.array(consumes)
        self.produces = np.array(produces)
        self.color = np.array(material)
        self.can_reproduce = False
        self.step_alive = 0
        self.policy = policy
        self.is_active = True
        self.consumed_material = False
        self.consumed_energy = False

    def mutate(self):
        self.state.p_pos += np.random.uniform(-0.1, 0.1, 2)
        self.color += np.random.uniform(-0.1, 0.1, 3)
        self.consumes += np.random.uniform(-0.1, 0.1, 3)
        self.produces += np.random.uniform(-0.1, 0.1, 3)
        self.policy.mutate()

class PetriEnergyAgent(Agent):

    def __init__(self, config, loc, consumes, produces, material, policy=None):
        super().__init__()
        self.state.p_pos = np.array(loc)
        self.consumes = np.array(consumes)
        self.produces = np.array(produces)
        self.color = np.array(material)
        self.policy = policy
        self.is_active = True
        self.consumed_material = False
        self.max_energy = config['max_energy']
        self.energy_store = self.max_energy / 2
        self.move_cost = config['move_cost']
        self.idle_cost = config['idle_cost']
        self.prod_cost = config['prod_cost']
        self.reprod_cost = config['reprod_cost']

    def mutate(self):
        self.state.p_pos += np.random.uniform(-0.1, 0.1, 2)
        self.color += np.random.uniform(-0.1, 0.1, 3)
        self.consumes += np.random.uniform(-0.1, 0.1, 3)
        self.produces += np.random.uniform(-0.1, 0.1, 3)
        self.policy.mutate()



class PetriWorld(World):

    def __init__(self, config):
        super().__init__()
        self.world_bound = config['world_bound']
        self.eating_distace = config['eating_distance']

    @property
    def agent_positions(self):
        return [agent.state.p_pos for agent in self.agents]

    @property
    def active_resource_positions(self):
        return [resource.state.p_pos for resource in self.landmarks if resource.is_active]

    @property
    def active_resources(self):
        return [resource for resource in self.landmarks if resource.is_active]

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
            if entity.state.p_pos[0] > self.world_bound:
                entity.state.p_pos[0] = -self.world_bound
            elif entity.state.p_pos[0] < -self.world_bound:
                entity.state.p_pos[0] = self.world_bound

            if entity.state.p_pos[1] > self.world_bound:
                entity.state.p_pos[1] = -self.world_bound
            elif entity.state.p_pos[1] < -self.world_bound:
                entity.state.p_pos[1] = self.world_bound
    