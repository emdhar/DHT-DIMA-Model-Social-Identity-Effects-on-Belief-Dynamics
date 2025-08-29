
import numpy as np
from AgentRoleBelief import AgentRoleManager, BeliefEngine

class BasicDHT:
    def __init__(self, N, M, W, distributions, conspirators_idx=None, debunkers_idx=None, seed=None, dev = None):
        self.N = N
        self.M = M
        self.W = W
        self.roles = AgentRoleManager(N, conspirators_idx, debunkers_idx)
        self.beliefs = BeliefEngine(N, M, distributions)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        if dev is None:
            self.dev = 0
        else:
            self.dev = dev

    def get_params(self):
        """Return current parameter configuration"""
        return {'N': self.N, 'M': self.M, 'seed': self.seed,'conspirators_idx': list(self.roles.conspirators),'debunkers_idx': list(self.roles.debunkers)}

    def step(self):
        # X = [self.beliefs.distributions[i][self.M - 1].rvs() for i in range(self.N)]
        X = [self.beliefs.distributions[i][self.M - 1].rvs() + np.random.normal(0, self.dev) for i in range(self.N)]
        self.beliefs.observe_and_update_public_beliefs(X, self.roles.conspirators, self.roles.debunkers)
        # print("W:   ", self.W)
        self.beliefs.update_private_beliefs(self.W, self.roles.regulars)

    def run(self, T):
        history = np.zeros((T + 1, self.N, self.M))
        history[0] = self.beliefs.q.copy()
        for t in range(1, T + 1):
            self.step()
            history[t] = self.beliefs.q.copy()
        return history

    
    def export_run_data(self, filename):
        data = {
            'final_beliefs': self.beliefs.q.copy(),
            'parameters': self.get_params(),
            'final_truth_belief': self.beliefs.q[:, -1].mean(),
            'polarization': np.var(self.beliefs.q[:, -1])
        }
        np.save(filename, data)
        return data