import numpy as np
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.utils.agent_selector import agent_selector
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
        self.color = np.array([255, 255, 0])


class PetriMaterial(PetriLandmark):

    def __init__(self, loc, color):
        super().__init__()
        self.resource_type = "material"
        self.state.p_pos = np.array(loc)
        self.color = np.array(color)


class PetriAgent(Agent):

    def __init__(self, config, loc, consumes, produces, material, policy=None):
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
        self.color += np.random.uniform(-0.05, 0.05, 3)
        self.consumes += np.random.uniform(-0.05, 0.05, 3)
        self.produces += np.random.uniform(-0.05, 0.05, 3)

        self.color[self.color < 0] = 0
        self.color[self.color > 1] = 1
        self.consumes[self.consumes < 0] = 0
        self.consumes[self.consumes > 1] = 1
        self.produces[self.produces < 0] = 0
        self.produces[self.produces > 1] = 1

        self.policy.mutate()

class PetriEnergyAgent(PetriAgent):

    def __init__(self, config, loc, consumes, produces, material, policy=None):
        super().__init__(config, loc, consumes, produces, material, policy=policy)
        self.max_energy = config['max_energy']
        self.energy_store = self.max_energy / 2
        self.move_cost = config['move_cost']
        self.idle_cost = config['idle_cost']
        self.prod_cost = config['prod_cost']
        self.reprod_cost = config['reprod_cost']
        self.currently_eating = None
        self.currently_attacking = None

    def can_produce_resource(self):
        if self.energy_store - self.prod_cost > 0:
            self.energy_store -= self.prod_cost
            return True
        else:
            return False

    def idle(self):
        self.energy_store -= self.idle_cost

    def move(self):
        self.energy_store -= self.move_cost

    def assign_eat(self, idx, world):
        self.currently_eating = world.landmarks[idx]

    def eat(self, landmark):
        dist = np.sum(np.square(self.consumes - landmark.color))
        # Maximum distance is 3 since all colors entries are within [0, 1]
        # Highest energy it can consume is max_energy / 2
        new_energy = (3 - dist) / 3 * self.max_energy / 2
        self.energy_store = min(self.max_energy, self.energy_store + new_energy)

    def reproduce(self):
        if self.energy_store - self.reprod_cost > 0:
            self.energy_store -= self.reprod_cost
            return True
        else:
            return False

    def assign_attack(self, agent):
        self.currently_attacking = agent

    def attack_agent(self, ag):
        # TODO: Fix this stuff
        self.energy_store = min(self.max_energy, self.energy_store + 300)

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
    