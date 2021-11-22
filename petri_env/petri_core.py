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
        self.is_waste = False


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
        self.color = np.clip(self.color + np.random.uniform(-0.05, 0.05, 3), 0, 1)
        self.consumes = np.clip(self.consumes + np.random.uniform(-0.05, 0.05, 3), 0, 1)
        self.produces = np.clip(self.produces + np.random.uniform(-0.05, 0.05, 3), 0, 1)
        self.energy_store = self.max_energy / 3
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
        self.energy_store = self.max_energy / 3
        self.move_cost = config['move_cost']
        self.idle_cost = config['idle_cost']
        self.prod_cost = config['prod_cost']
        self.reprod_cost = config['reprod_cost']
        self.attack_cost = config['attack_cost']
        self.currently_eating = None
        self.currently_attacking = None
        self.lineage_length = 1

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
        self.currently_eating = world.active_resources[idx]

    def eat(self, landmark):
        dist = np.sum(np.abs(self.consumes - landmark.color))
        # Maximum distance is 3 since all colors entries are within [0, 1]
        # Highest energy it can consume is max_energy / 2
        new_energy = (3 - dist) / 3 * self.max_energy / 3
        self.energy_store = min(self.max_energy, self.energy_store + new_energy)

    def reproduce(self):
        if self.energy_store - self.reprod_cost > 0:
            self.energy_store -= self.reprod_cost
            return True
        else:
            return False

    def attack(self):
        if self.energy_store - self.attack_cost < 0:
            self.energy_store -= self.attack_cost
            return True
        else:
            self.idle()
            return False

    def assign_attack(self, agent):
        self.currently_attacking = agent

    def attack_agent(self, ag):
        dist = np.sum(np.square(self.consumes - ag.color))
        # Maximum distance is 3 since all colors entries are within [0, 1]
        # Highest energy it can consume is max_energy / 2
        new_energy = (3 - dist) / 3 * self.max_energy / 2
        self.energy_store = min(self.max_energy, self.energy_store + new_energy)


class PetriWorld(World):

    def __init__(self, config):
        super().__init__()
        self.world_bound = config['world_bound']
        self.eating_distace = config['eating_distance']

    @property
    def active_resource_positions(self):
        return [resource.state.p_pos for resource in self.landmarks if resource.is_active]

    @property
    def active_resources(self):
        return [resource for resource in self.landmarks if resource.is_active]

    @property
    def resource_positions(self):
        return [resource.state.p_pos for resource in self.landmarks]

    @property
    def agent_positions(self):
        return [agent.state.p_pos for agent in self.agents]

    def calculate_distances(self):
        self.agent_res_distances = euclidean_distances(self.agent_positions, self.active_resource_positions)
        self.agent_agent_distances = euclidean_distances(self.agent_positions, self.agent_positions)
        np.fill_diagonal(self.agent_agent_distances, np.inf)

        agent_positions, agent_velocities, all_consumes, all_produces, all_a_colors = [], [], [], [], []
        
        for agent in self.agents:
            agent_positions.append(agent.state.p_pos)
            agent_velocities.append(agent.state.p_vel)
            all_consumes.append(agent.consumes)
            all_produces.append(agent.produces)
            all_a_colors.append(agent.color)

        agent_positions = np.array(agent_positions) / self.world_bound
        agent_velocities = np.array(agent_velocities)
        all_consumes = np.array(all_consumes)
        all_produces = np.array(all_produces)
        all_a_colors = np.array(all_a_colors)

        landmark_positions, landmark_colors = [], []
        for landmark in self.active_resources:
            landmark_positions.append(landmark.state.p_pos)
            landmark_colors.append(landmark.color)
        
        landmark_positions = np.array(landmark_positions) / self.world_bound
        landmark_colors = np.array(landmark_colors)

        agent_landmark_dists = np.expand_dims(agent_positions, 1) - landmark_positions
        self.agent_landmark_feats = np.concatenate([agent_landmark_dists, 
                                                        np.tile(landmark_colors, (agent_positions.shape[0], 1, 1))], 
                                                    axis=2)

        agent_agent_dists = np.expand_dims(agent_positions, 1) - agent_positions
        
        self.agent_agent_feats = np.concatenate([agent_agent_dists, 
                                                        np.tile(agent_velocities, (agent_velocities.shape[0], 1, 1)),
                                                        np.tile(all_consumes, (all_consumes.shape[0], 1, 1)),
                                                        np.tile(all_produces, (all_produces.shape[0], 1, 1)),
                                                        np.tile(all_a_colors, (all_a_colors.shape[0], 1, 1))], 
                                                    axis=2)
        print(self.agent_agent_feats.shape)
        
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
            """
            if entity.state.p_pos[0] > self.world_bound:
                entity.state.p_pos[0] = -self.world_bound
            elif entity.state.p_pos[0] < -self.world_bound:
                entity.state.p_pos[0] = self.world_bound

            if entity.state.p_pos[1] > self.world_bound:
                entity.state.p_pos[1] = -self.world_bound
            elif entity.state.p_pos[1] < -self.world_bound:
                entity.state.p_pos[1] = self.world_bound
            """
            
            if entity.state.p_pos[0] > self.world_bound:
                entity.state.p_pos[0] = self.world_bound
                #entity.energy_store = -1
            elif entity.state.p_pos[0] < -self.world_bound:
                entity.state.p_pos[0] = -self.world_bound
                #entity.energy_store = -1

            if entity.state.p_pos[1] > self.world_bound:
                entity.state.p_pos[1] = self.world_bound
                #entity.energy_store = -1
            elif entity.state.p_pos[1] < -self.world_bound:
                entity.state.p_pos[1] = -self.world_bound
                #entity.energy_store = -1
            