from hashlib import new
import numpy as np
from petri_env.petri_core import PetriEnergy, PetriMaterial

class ResourceGenerator():

    def __init__(self, config, world):
        self.world = world
        self.recov_time = config['recov_time']
        self.world_bound = config['world_bound']
        self.use_energy_resource = config['use_energy_resource']

    def generate_initial_resources(self):
        raise NotImplementedError

    def update_resources(self):
        for landmark in self.world.landmarks:
            if not landmark.is_active:
                landmark.inactive_count += 1
                if landmark.inactive_count >= self.recov_time:
                    self.activate_resource(landmark)

    def activate_resource(self, resource):
        raise NotImplementedError

class RandomResourceGenerator(ResourceGenerator):

    def __init__(self, config, world):
        super().__init__(config, world)
        self.num_resources = config['num_resources']

    def generate_initial_resources(self):
        while len(self.world.landmarks) < self.num_resources:
            resource_kind = np.random.choice(2, 1)
            loc = np.random.uniform(-self.world_bound, self.world_bound, 2)
            if resource_kind == 0 and self.use_energy_resource:
                # Energy
                new_resource = PetriEnergy(loc)
            else:
                color = np.random.uniform(0, 1, 3)
                new_resource = PetriMaterial(loc, color)

            self.world.landmarks.append(new_resource)
            self.activate_resource(new_resource)

    def activate_resource(self, resource):
        new_loc = np.random.uniform(-self.world_bound, self.world_bound, 2)
        resource.loc = new_loc
        resource.is_active = True
        resource.inactive_count = 0


class FixedResourceGenerator(ResourceGenerator):

    def __init__(self, config, world):
        super().__init__(config, world)

    def generate_initial_resources(self, locs, types):
        return 

    def activate_resource(self, resource):
        resource.is_active = True
        resource.inactive_count = 0