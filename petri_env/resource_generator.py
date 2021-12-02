import numpy as np
from petri_env.petri_core import PetriMaterial
from numpy.random import multivariate_normal

class ResourceGenerator():

    def __init__(self, config, world):
        self.world = world
        self.recov_time = config['recov_time']
        self.world_bound = config['world_bound']

    def generate_initial_resources(self):
        raise NotImplementedError

    def update_resources(self):
        res_to_keep = [True for _ in range(len(self.world.landmarks))]
        for i, landmark in enumerate(self.world.landmarks):
            if not landmark.is_active:
                if landmark.is_waste:
                    res_to_keep[i] = False
                else:
                    landmark.inactive_count += 1
                    if landmark.inactive_count >= self.recov_time:
                        self.activate_resource(landmark)
        self.world.landmarks = [landmark for i, landmark in enumerate(self.world.landmarks) if res_to_keep[i]]

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
            #color = np.random.uniform(0, 1, 3)
            color = np.array([1., 0., 0.])
            new_resource = PetriMaterial(loc, color)

            self.world.landmarks.append(new_resource)
            self.activate_resource(new_resource)

    def activate_resource(self, resource):
        new_loc = np.random.uniform(-self.world_bound, self.world_bound, 2)
        resource.state.p_pos = new_loc
        resource.is_active = True
        resource.inactive_count = 0

class BimodalResourceGenerator(ResourceGenerator):

    def __init__(self, config, world):
        super().__init__(config, world)
        self.num_resources = config['num_resources']

    def generate_initial_resources(self):
        while len(self.world.landmarks) < self.num_resources:
            resource_kind = np.random.choice(2, 1)
            loc = np.random.uniform(-self.world_bound, self.world_bound, 2)
            if resource_kind == 0:
                mean = np.array([self.world_bound / 2, self.world_bound / 2])
                color = np.array([1., 0., 0.])
            else:
                mean = np.array([-self.world_bound / 2, -self.world_bound / 2])
                color = np.array([0., 0., 1.])
            loc = multivariate_normal(mean, np.array([[4, 1], [1, 4]]))
            new_resource = PetriMaterial(loc, color)
            self.world.landmarks.append(new_resource)
            self.activate_resource(new_resource)

    def activate_resource(self, resource):
        resource.is_active = True
        resource.inactive_count = 0