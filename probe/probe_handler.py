from probe import Probe

class ProbeHandler():
    def __init__(self, num_state_variables, is_supervised = False):
        self.num_state_variables = num_state_variables
        self.is_supervised = is_supervised

        self.probes = {}
    
    def setup_probes(self):
        if self.is_supervised:
            print('Implement fully supervised probe')
        else:
            for i in range(self.num_state_variables):
                self.probes[i] = Probe()

    def train(self):
        self.setup_probes()