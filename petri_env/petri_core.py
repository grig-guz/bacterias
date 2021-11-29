import numpy as np
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.utils.agent_selector import agent_selector
from utils import *
from neat.genome import DefaultGenome
import neat

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
        self.step_alive = 0
        self.policy = policy
        self.is_active = True
        self.consumed_material = False
        self.sigma = config['sigma']

    def mutate(self):
        self.color = np.clip(self.color + np.random.uniform(-self.sigma, self.sigma, 3), 0, 1)
        self.consumes = np.clip(self.consumes + np.random.uniform(-self.sigma, self.sigma, 3), 0, 1)
        self.produces = np.clip(self.produces + np.random.uniform(-self.sigma, self.sigma, 3), 0, 1)
        self.energy_store = self.max_energy / 3
        self.consumed_material = False
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
        self.energy_distance_type = config['energy_distance_type']
        self.species_eating_cutoff = config['species_eating_cutoff']
        self.currently_eating = None
        self.currently_attacking = None
        self.lineage_length = 1

    def can_produce_resource(self):
        energy_gain = self.get_energy_gain(self.produces)
        if self.energy_store - energy_gain > 0 and self.consumed_material:
            self.energy_store -= energy_gain
            return True
        else:
            self.idle()
            return False

    def get_energy_gain(self, color):
        if self.energy_distance_type == 'l1':
            dist = np.sum(np.abs(self.consumes - color))
            # Maximum distance is 3 since all colors entries are within [0, 1]
            # Highest energy it can consume is max_energy / 2
            new_energy = (3 - dist) / 3 * self.max_energy / 2
        elif self.energy_distance_type == 'exp':
            dist = np.exp(-np.sum(np.square(self.consumes - color))*2)
            new_energy = dist * self.max_energy

        return new_energy

    def idle(self):
        self.energy_store -= self.idle_cost

    def move(self):
        self.energy_store -= self.move_cost

    def assign_eat(self, idx, world):
        self.currently_eating = world.active_resources[idx]

    def eat(self, landmark):
        new_energy = self.get_energy_gain(landmark.color)
        self.energy_store = min(self.max_energy, self.energy_store + new_energy)
        self.consumed_material = True

    def reproduce(self):
        if self.energy_store - self.reprod_cost > 0:
            self.energy_store -= self.reprod_cost
            return True
        else:
            self.idle()
            return False

    def can_attack(self, agent2):
        if self.energy_store - self.attack_cost > 0 and self.agent_dist(agent2) > self.species_eating_cutoff:
            self.energy_store -= self.attack_cost
            return True
        else:
            self.idle()
            return False

    def agent_dist(self, agent2):
        dist = np.sum(np.abs(self.consumes - agent2.consumes) + \
                        np.abs(self.color - agent2.color) + \
                        np.abs(self.produces - agent2.produces))

        return dist / 3

    def assign_attack(self, agent2):
        self.currently_attacking = agent2

    def attack_agent(self, agent2):
        new_energy = self.get_energy_gain(agent2.color)
        self.energy_store = min(self.max_energy, self.energy_store + new_energy)
        self.consumed_material = True


class PetriNeatAgent(PetriEnergyAgent):

    def __init__(self, config, neat_config, loc, consumes, produces, material, policy=None):
        super().__init__(config, loc, consumes, produces, material, policy=policy)
        self.neat_config = neat_config
        self.genome = DefaultGenome(0)
        self.genome.fitness = 0
        self.genome.configure_new(self.neat_config.genome_config)
        self.policy = neat.nn.FeedForwardNetwork.create(self.genome, self.neat_config)

    def mutate(self):
        self.color = np.clip(self.color + np.random.uniform(-self.sigma, self.sigma, 3), 0, 1)
        self.consumes = np.clip(self.consumes + np.random.uniform(-self.sigma, self.sigma, 3), 0, 1)
        self.produces = np.clip(self.produces + np.random.uniform(-self.sigma, self.sigma, 3), 0, 1)
        self.energy_store = self.max_energy / 3
        self.consumed_material = False

        new_genome = DefaultGenome(0)
        new_genome.fitness = 0
        new_genome.configure_crossover(self.genome, self.genome, self.neat_config.genome_config)
        new_genome.mutate(self.neat_config.genome_config)
        self.genome = new_genome
        self.policy = neat.nn.FeedForwardNetwork.create(self.genome, self.neat_config)


class PetriWorld(World):

    def __init__(self, config):
        super().__init__()
        self.world_bound = config['world_bound']
        self.eating_distace = config['eating_distance']
        self.wrap_around = config['wrap_around']
        start = -self.world_bound
        bins = []
        while start < self.world_bound:
            bins.append(start)
            start += 0.1
        self.bins = np.array(bins)
        self.cell_size = int(self.world_bound * 2 / 0.1)
        self.cell = np.zeros(shape=(14, self.cell_size, self.cell_size))

    def calculate_distances(self):
        if len(self.agents) > 0:
            self.active_resources = [resource for resource in self.landmarks if resource.is_active]
            self.active_resource_positions = [resource.state.p_pos for resource in self.landmarks if resource.is_active]
            self.agent_positions = [agent.state.p_pos for agent in self.agents]
            self.agent_res_distances = euclidean_distances(self.agent_positions, self.active_resource_positions)
            self.agent_agent_distances = euclidean_distances(self.agent_positions, self.agent_positions)
        
            np.fill_diagonal(self.agent_agent_distances, np.inf)
        """
        self.cell = -np.ones(shape=(self.cell.shape))
        for agent in self.agents:
            x, y = agent.state.p_pos
            x = np.digitize(x, self.bins) - 1
            y = np.digitize(-y, self.bins) - 1
            self.cell[:3, y, x] = agent.color
            self.cell[3:6, y, x] = agent.consumes
            self.cell[6:9, y, x] = agent.produces
        
        for landmark in self.active_resources:
            x, y = landmark.state.p_pos
            x = np.digitize(x, self.bins) - 1
            y = np.digitize(-y, self.bins) - 1
            self.cell[9:12, y, x] = landmark.color
        """
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
        #p_force = self.apply_environment_force(p_force)
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
            if self.wrap_around:
                if entity.state.p_pos[0] > self.world_bound:
                    entity.state.p_pos[0] = -self.world_bound
                elif entity.state.p_pos[0] < -self.world_bound:
                    entity.state.p_pos[0] = self.world_bound

                if entity.state.p_pos[1] > self.world_bound:
                    entity.state.p_pos[1] = -self.world_bound
                elif entity.state.p_pos[1] < -self.world_bound:
                    entity.state.p_pos[1] = self.world_bound
            else:
                if entity.state.p_pos[0] > self.world_bound:
                    entity.state.p_pos[0] = self.world_bound
                    #entity.energy_store = -1
                if entity.state.p_pos[0] < -self.world_bound:
                    entity.state.p_pos[0] = -self.world_bound
                    #entity.energy_store = -1

                if entity.state.p_pos[1] > self.world_bound:
                    entity.state.p_pos[1] = self.world_bound
                    #entity.energy_store = -1
                if entity.state.p_pos[1] < -self.world_bound:
                    entity.state.p_pos[1] = -self.world_bound
                    #entity.energy_store = -1
