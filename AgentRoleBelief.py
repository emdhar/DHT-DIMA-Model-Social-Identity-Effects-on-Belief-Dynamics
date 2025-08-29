import numpy as np

class AgentRoleManager:
    def __init__(self, N, conspirators_idx=None, debunkers_idx=None):
        self.N = N
        self.conspirators = set(conspirators_idx) if conspirators_idx is not None else set()
        self.debunkers = set(debunkers_idx) if debunkers_idx is not None else set()
        self.regulars = set(range(N)) - self.conspirators - self.debunkers
        self.active_identities = [('personal', None)] * N


class BeliefEngine:
    def __init__(self, N, M, distributions):
        self.N = N
        self.M = M
        self.distributions = distributions
        self.q = np.full((N, M), 1.0 / M)  # private beliefs
        self.b = np.zeros((N, M))         # public beliefs

    def observe_and_update_public_beliefs(self, X, conspirators, debunkers):
        for i in range(self.N):
            if i in conspirators:
                self.b[i] = np.eye(self.M)[0]
            elif i in debunkers:
                self.b[i] = np.eye(self.M)[self.M - 1]
            else:
                likelihoods = np.array([self.distributions[i][k].pdf(X[i]) for k in range(self.M)])
                unnorm = likelihoods * self.q[i]
                self.b[i] = unnorm / (np.sum(unnorm) + 1e-9)

    def update_private_beliefs(self, W, regulars):
        # print("update_private_beliefs")
        log_b = np.log(self.b + 1e-20)
        new_q = self.q.copy()
        for i in regulars:
            agg_log_b = np.dot(W[i], log_b)
            exp_agg = np.exp(agg_log_b - np.max(agg_log_b))
            new_q[i] = exp_agg / np.sum(exp_agg)
        self.q = new_q