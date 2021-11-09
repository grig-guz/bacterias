from hashlib import new
import numpy as np
from petri_env.petri_core import PetriEnergy, PetriMaterial

class ResourceGenerator():

    def __init__(self, world, recov_time):
        self.world = world
        self.recov_time = recov_time

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

    def __init__(self, world, recov_time, num_resources):
        super().__init__(world, recov_time)
        self.num_resources = num_resources

    def generate_initial_resources(self):
        while len(self.world.landmarks) < self.num_resources:
            resource_kind = np.random.choice(2, 1)
            if resource_kind == 0:
                # Energy
                new_resource = PetriEnergy(None)
            else:
                new_color = np.random.uniform(0, 1, 3)
                new_resource = PetriMaterial(None, new_color)

            self.world.landmarks.append(new_resource)
            self.activate_resource(new_resource)

    def activate_resource(self, resource):
        new_loc = np.random.uniform(-5, 5, 2)
        resource.loc = new_loc
        resource.is_active = True
        resource.inactive_count = 0


class FixedResourceGenerator(ResourceGenerator):

    def __init__(self, world, recov_time):
        super().__init__(world, recov_time)

    def generate_initial_resources(self, locs, types):
        return 

    def activate_resource(self, resource):
        resource.is_active = True
        resource.inactive_count = 0