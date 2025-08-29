
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from AgentRoleBelief import AgentRoleManager, BeliefEngine
from IdentitySalience import ClusteringEngine, IdentityEngine, WeightAdjuster

class DHTwdDIMA:
    def __init__(self, N, M, G, W, distributions, agent_features, themes, normative_groups,
                 conspirators_idx=None, debunkers_idx=None, dima_beta=None, seed = 42, app_factor=None, salience_threshold = None, dev = None, logger = None):
        self.N, self.M, self.G, self.W = N, M, G, W
        self.themes = themes
        self.current_theme = "politics"
        self.roles = AgentRoleManager(N, conspirators_idx, debunkers_idx)
        self.clusterer = ClusteringEngine(G, agent_features, themes)
        if salience_threshold is None:
            self.identity = IdentityEngine(agent_features, normative_groups, N, self.clusterer.max_clusters)
        else:
            self.identity = IdentityEngine(agent_features, normative_groups, N, self.clusterer.max_clusters, salience_threshold = salience_threshold)
        self.logger = logger

        if dev is None:
            self.dev = 0
        else:
            self.dev = dev
        self.beliefs = BeliefEngine(N, M, distributions)
        self.weight_adjuster = WeightAdjuster(G, W, dima_beta)
        self.seed = seed
        self.app_factor = 0.5 if app_factor is None else app_factor
        np.random.seed(seed)

        
        # Add tracking for experiments
        self.q_history = []
        self.identity_history = []
        
    def get_params(self):
        """Return current parameter configuration"""
        return {
            'N': self.N, 'M': self.M, 'seed': self.seed,
            'dima_beta': getattr(self.weight_adjuster, 'dima_beta', None),
            'salience_threshold': getattr(self.identity, 'salience_threshold', None),
            'current_theme': self.current_theme
        }

    def step(self, t, T):
        clusters_cache = {}
        for i in self.roles.regulars:
            result = self.clusterer.perceive_and_cluster(i, self.current_theme)
            #print("result", result)
            if result is None:
                self.roles.active_identities[i] = ('personal', None)
                clusters_cache[i] = (None, None, []) # no clusters/context to reuse
                continue
            clusters, _ = result
            context = [idx for cluster in clusters.values() for idx in cluster]
            
            # Social approval 
            id_i = self.roles.active_identities[i]
            if id_i == ('personal', None):
                approval = 0.0
            else:
                same = 0
                total = 0
                for j in self.G.neighbors(i):
                    total += 1
                    if self.roles.active_identities[j] == id_i:
                        same += 1
                approval = (same / total) if total > 0 else 0.0
                # if approval > 0:
                #     print("===========Approval================", approval)
            
            # Accuracy 
            true_idx = self.M - 1
            accuracy = float(self.beliefs.q[i, true_idx])
            # print("Accuracy", accuracy)

            # Blend and map to [0.5, 1.0]
            score = self.app_factor * approval + (1 - self.app_factor)  * accuracy
            outcome_factor = 0.5 + 0.5 * score
            # print("outcome_factor",outcome_factor )

            chosen_identity, cache_tuple = self.identity.update_identity(i, clusters, context, self.roles.active_identities[i], outcome_factor = outcome_factor)
            self.roles.active_identities[i] = chosen_identity
            clusters_cache[i] = cache_tuple  # (clusters, centers placeholder, context)

        # X = [self.beliefs.distributions[i][self.M - 1].rvs() for i in range(self.N)]

        X = [self.beliefs.distributions[i][self.M - 1].rvs() + np.random.normal(0, self.dev) for i in range(self.N)]


        self.beliefs.observe_and_update_public_beliefs(X, self.roles.conspirators, self.roles.debunkers)
        W_dyn = self.weight_adjuster.adjust(self.roles.regulars, self.roles.active_identities, clusters_cache)
        # print("W_dyn:   ", W_dyn)
        # y = (W_dyn == self.W).all()
        # print("Weight not changed: ===========", y, "===============")
        self.beliefs.update_private_beliefs(W_dyn, self.roles.regulars)


            # Track data for experiments
        self.q_history.append(self.beliefs.q.copy())
        identity_types = [id_type for id_type, _ in self.roles.active_identities]
        self.identity_history.append({
            'personal': identity_types.count('personal') / len(identity_types),
            'dynamic': identity_types.count('dynamic') / len(identity_types),
            'normative': identity_types.count('normative') / len(identity_types)
        })

    # def run(self, T, theme_schedule=None):
    #     history = np.zeros((T + 1, self.N, self.M))
    #     history[0] = self.beliefs.q.copy()
    #     for t in range(1, T + 1):
    #         if theme_schedule:
    #             self.current_theme = theme_schedule(t)
    #         self.step(t, T)
    #         history[t] = self.beliefs.q.copy()
    #     return history


    def run(self, T, theme_schedule=None, run_id=None, output_dir="logs"):
        
        if self.logger is None and run_id is not None:
            from logger import SimulationLogger
            self.logger = SimulationLogger(run_id, output_dir)
    
        history = np.zeros((T + 1, self.N, self.M))
        history[0] = self.beliefs.q.copy()
    
        for t in range(1, T + 1):
            if theme_schedule:
                self.current_theme = theme_schedule(t)
    
            self.step(t, T)
            history[t] = self.beliefs.q.copy()
    
            if self.logger:
                self.logger.log_step(t, self.beliefs.q, self.get_identity_states())
                for i in self.roles.regulars:
                    evidence_strength = self.get_evidence_strength(i)
                    belief_before = history[t - 1][i]
                    belief_after = history[t][i]

                    # belief_before = self.q_history[t - 1][i]
                    # belief_after = self.q_history[t][i]
    
                    if evidence_strength[-1] > 2.0 and belief_after[-1] < belief_before[-1]:
                        context = {
                            "dima_beta": self.weight_adjuster.dima_beta,
                            #"salience": self.identity.salience[i],
                            "local_consensus": self.get_local_consensus(i),
                            "signal_strength": evidence_strength[-1],
                            "network_position": self.get_network_metrics(i)
                        }
                        self.logger.log_reversal_event(i, t, evidence_strength[-1],
                                                       belief_before[-1], belief_after[-1], context)
    
        if self.logger:
            self.logger.finalize()

        return history
   
    def get_identity_states(self):
        return np.array([1 if id_type != 'personal' else 0 for id_type, _ in self.roles.active_identities])

    def get_evidence_strength(self, agent_id):
        return [self.beliefs.distributions[agent_id][k].mean for k in range(self.M)]
    
    def get_local_consensus(self, agent_id):
        neighbors = list(self.G.neighbors(agent_id))
        if not neighbors:
            return 0.0
        neighbor_beliefs = self.beliefs.q[neighbors, -1]  # belief in truth
        return float(np.mean(neighbor_beliefs))
    
    def get_network_metrics(self, agent_id):
        import networkx as nx
        clustering = nx.clustering(self.G, agent_id)
        centrality = nx.degree_centrality(self.G)[agent_id]
        return {"clustering": clustering, "centrality": centrality}

    
    def export_run_data(self, filename):
        data = {
            'q_history': np.array(self.q_history),
            'identity_history': self.identity_history,
            'parameters': self.get_params(),
            'final_truth_belief': np.array(self.q_history)[-1, :, -1].mean(),
            'polarization': np.var(np.array(self.q_history)[-1, :, -1])
        }
        np.save(filename, data)
        return data
