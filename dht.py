
import numpy as np
from AgentRoleBelief import AgentRoleManager, BeliefEngine

class BasicDHT:
    def __init__(self, N, M, W, distributions, conspirators_idx=None, debunkers_idx=None, seed=None, dev = None, logger=None):
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

        self.logger = logger

    def get_params(self):
        """Return current parameter configuration"""
        return {'N': self.N, 'M': self.M, 'seed': self.seed,'conspirators_idx': list(self.roles.conspirators),'debunkers_idx': list(self.roles.debunkers)}

    def step(self):
        # X = [self.beliefs.distributions[i][self.M - 1].rvs() for i in range(self.N)]
        X = [self.beliefs.distributions[i][self.M - 1].rvs() + np.random.normal(0, self.dev) for i in range(self.N)]
        self.beliefs.observe_and_update_public_beliefs(X, self.roles.conspirators, self.roles.debunkers)
        # print("W:   ", self.W)
        self.beliefs.update_private_beliefs(self.W, self.roles.regulars)
        if self.logger:
            self.logger.log_step(t, self.beliefs.q, self.get_identity_states())
            for i in self.roles.regulars:
                evidence_strength = self.get_evidence_strength(i)
                belief_before = history[t - 1][i]
                belief_after = self.beliefs.q[i]

                if evidence_strength[-1] > 2.0 and belief_after[-1] < belief_before[-1]:
                    context = {
                        "signal_strength": evidence_strength[-1],
                        "network_position": self.get_network_metrics(i)
                    }
                    self.logger.log_reversal_event(i, t, evidence_strength[-1],
                                                   belief_before[-1], belief_after[-1], context)

    def run(self, T , run_id=None, output_dir="logs"):
        if self.logger is None and run_id is not None:
            from logger import SimulationLogger
            self.logger = SimulationLogger(run_id, output_dir)
        history = np.zeros((T + 1, self.N, self.M))
        history[0] = self.beliefs.q.copy()
        for t in range(1, T + 1):
            self.step()
            history[t] = self.beliefs.q.copy()
        if self.logger:
            self.logger.finalize()
        return history

    def get_identity_states(self):
        # BasicDHT doesn't use identity types, so default to all personal
        return np.zeros(self.N)

    def get_evidence_strength(self, agent_id):
        return [self.beliefs.distributions[agent_id][k].mean for k in range(self.M)]

    def get_network_metrics(self, agent_id):
        import networkx as nx
        clustering = nx.clustering(self.roles.G, agent_id) if hasattr(self.roles, 'G') else 0.0
        centrality = nx.degree_centrality(self.roles.G)[agent_id] if hasattr(self.roles, 'G') else 0.0
        return {"clustering": clustering, "centrality": centrality}


    
    def export_run_data(self, filename):
        data = {
            'final_beliefs': self.beliefs.q.copy(),
            'parameters': self.get_params(),
            'final_truth_belief': self.beliefs.q[:, -1].mean(),
            'polarization': np.var(self.beliefs.q[:, -1])
        }
        np.save(filename, data)
        return data