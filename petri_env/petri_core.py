from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World


class PetriEnergy(Landmark):

    def __init__(self):
        super.__init__()
        # TODO fix this
        self.color = 'yellow'
        self.energy_val = 10


class PetriMaterial(Landmark):

    def __init__(self):
        super.__init__()
        # TODO fix this
        self.color = 'thecolor'

class PetriAgent(Agent):

    def __init__(self):
        super().__init__()

        # TODO: Fix this
        self.consumes = []
        self.produces = []
        self.color = "agentcolor"


class PetriWorld(World):

    def __init__(self):
        super().__init__()