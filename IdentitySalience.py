
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

class ClusteringEngine:
    def __init__(self, G, agent_features, themes, max_clusters=5):
        self.G = G
        self.agent_features = agent_features
        self.themes = themes
        self.max_clusters = max_clusters

    def find_optimal_k(self, features):
        if len(features) <= 2:
            return 1
        best_k, best_score = 2, -1
        for k in range(2, min(self.max_clusters + 1, len(features))):
            kmeans = KMeans(n_clusters=k, random_state=10, n_init='auto').fit(features)
            score = silhouette_score(features, kmeans.labels_)
            if score > best_score:
                best_k, best_score = k, score
        #print("score: {}, best_score: {}, best_k: {}".format(score, best_score, best_k))
        return best_k

    def perceive_and_cluster(self, agent_id, current_theme):
        one_hop = list(self.G.neighbors(agent_id))
        two_hop = [n for neighbor in one_hop for n in self.G.neighbors(neighbor)]
        context_ids = list(set([agent_id] + one_hop + two_hop))
        if len(context_ids) <= 1:
            return None
        theme_indices = self.themes[current_theme]
        context_features = self.agent_features[context_ids][:, theme_indices]  #picks up the context features corresponding to the theme_indices
        if context_features.shape[1] == 0:
            return None
        k = self.find_optimal_k(context_features)
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(context_features)
        clusters = {label: [] for label in range(k)}
        for i, idx in enumerate(context_ids):
            clusters[kmeans.labels_[i]].append(idx)
        #print("clusters: {}, cluster_centers_: {}".format(clusters, kmeans.cluster_centers_))
        return clusters, kmeans.cluster_centers_

class IdentityEngine:
    def __init__(self, agent_features, normative_groups, N, max_clusters, salience_threshold=2):
        self.agent_features = agent_features
        self.normative_groups = normative_groups
        self.salience_threshold = salience_threshold
        self.dynamic_accessibility = np.ones((N, max_clusters))
        self.normative_accessibility = np.ones((N, len(normative_groups)))

    def _closest_normative(self, centroid):
        names = list(self.normative_groups.keys())
        if len(names) == 0:
            return None, None
        dists = [np.linalg.norm(centroid - self.normative_groups[n]) for n in names]
        j = int(np.argmin(dists))
        return names[j], dists[j]
        
    #ensure normative_accessibility has a column for each normative group
    def _ensure_normative_capacity(self, N):
        current_cols = self.normative_accessibility.shape[1]
        needed = len(self.normative_groups)
        if needed > current_cols:
            extra_cols = needed - current_cols
            extra = np.ones((N, extra_cols))  # default accessibility = 1.0
            self.normative_accessibility = np.hstack([self.normative_accessibility, extra])

    def calculate_meta_contrast_fit(self, group_members, context_members, cap=20.0):
        in_group = self.agent_features[group_members]
        out_group = list(set(context_members) - set(group_members))
        if not out_group or len(in_group) == 0:
            return 0
        out_group_features = self.agent_features[out_group]
        centroid = np.mean(in_group, axis=0)
        intra_dist = float(np.median([np.linalg.norm(f - centroid) for f in in_group]))
        inter_dist = float(np.median([np.linalg.norm(in_f - out_f) for in_f in in_group for out_f in out_group_features]))
        #print(f"inter_dist{inter_dist}, intra_dist{intra_dist}")
        ratio = inter_dist / (intra_dist + 1e-2)
        fit = float(np.clip(ratio, 0.0, cap))
        #print(f"[MetaContrast]  inter_dist={inter_dist:.4f}, intra_dist={intra_dist:.4f}, ratio={ratio:.4f}, fit={fit:.4f}")
        return fit

    def update_identity(self, agent_id, clusters, context, current_identity, promote_dynamic_to_normative=True, match_threshold=0.4, outcome_factor = 0.5):
        saliences, identities = [], []
        for cluster_idx, members in clusters.items():
            if agent_id in members:
                fit = self.calculate_meta_contrast_fit(members, context)
                acc = self.dynamic_accessibility[agent_id, cluster_idx]
                #print("fit: {}, acc: {}".format(fit, acc))
                saliences.append(fit * acc)
                identities.append(('dynamic', cluster_idx))
                
        for i, (name, prototype) in enumerate(self.normative_groups.items()):
            fit = 1.0 / (np.linalg.norm(self.agent_features[agent_id] - prototype) + 1e-6)
            acc = self.normative_accessibility[agent_id, i]
            saliences.append(fit * acc)
            identities.append(('normative', name))

        # print("Saliences and ID:", saliences, identities)

        if not saliences:
            #print("personal active 1")
            return ('personal', None)
            
        max_idx = np.argmax(saliences)
        best_sal = saliences[max_idx]
        chosen = identities[max_idx]
    
        # Promote dynamic cluster centroid to normative memory if no close match
        if promote_dynamic_to_normative and clusters is not None:
            # Identify focal agent's in-group (only consider the group the agent belongs to)
            for cluster_idx, members in clusters.items():
                if agent_id in members:
                    if len(members) == 0:
                        # print(f"[DEBUG] Skipping empty cluster {cluster_idx}")
                        continue

                     # Calculate centroid of this dynamic cluster
                    centroid = np.mean(self.agent_features[members], axis=0)
                     # Find closest existing normative prototype
                    match_name, dist = self._closest_normative(centroid)
                    if (match_name is None) or (dist is None) or (dist > match_threshold):
                        # If no match or it's too far, promote this cluster, Create a unique name for the new normative group
                        base_name = f"dyn_{cluster_idx}_agent{agent_id}"
                        new_name = base_name
                        suffix = 1
                        while new_name in self.normative_groups:
                            suffix += 1
                            new_name = f"{base_name}_{suffix}"
                        # Add to normative store
                        self.normative_groups[new_name] = centroid
                        # print("normative groups", self.normative_groups)
                        # Expand accessibility matrix to include the new group
                        self._ensure_normative_capacity(N=self.dynamic_accessibility.shape[0])
                        # Give the focal agent a slightly higher initial accessibility for this new group
                        j = list(self.normative_groups.keys()).index(new_name)
                        self.normative_accessibility[agent_id, j] = 1.5
                        # print(f"[DEBUG] Promoted new normative group: '{new_name}' (dist={dist:.4f})")

                    # else:
                    #     print(f"[DEBUG] Skipped promotion: dynamic group {cluster_idx} too close to '{match_name}' (dist={dist:.4f})")

                    break  # only the focal agent's own cluster is considered
    
        # Activate chosen identity if above threshold and perform Outcome-sensitive accessibility updates
        # print("Best Sal : {},  Threshold: {}".format(best_sal, self.salience_threshold))
        if best_sal > self.salience_threshold:
            id_type, id_val = chosen
            decay_base, boost_base = 0.6, 0.9
             # Stronger outcome_factor -> stronger decay of other options and larger boost for chosen
            decay = 1.0 - (1.0 - decay_base) * outcome_factor
            boost = boost_base * outcome_factor
            if id_type == 'dynamic':
                self.dynamic_accessibility[agent_id] *= decay
                self.dynamic_accessibility[agent_id, id_val] += boost
            else:
                self.normative_accessibility[agent_id] *= decay
                idx = list(self.normative_groups.keys()).index(id_val)
                self.normative_accessibility[agent_id, idx] += boost
            return chosen, (clusters, None, context)
    
        return ('personal', None), (clusters, None, context)
        
       #  if saliences[max_idx] > self.salience_threshold:
       #      id_type, id_val = identities[max_idx]
       #      decay, boost = 0.3, 0.9
       #      if id_type == 'dynamic':
       #          self.dynamic_accessibility[agent_id] *= decay
       #          self.dynamic_accessibility[agent_id, id_val] += boost
       #      else:
       #          self.normative_accessibility[agent_id] *= decay
       #          idx = list(self.normative_groups.keys()).index(id_val)
       #          self.normative_accessibility[agent_id, idx] += boost
       #      #print("identities", identities[max_idx])    
       #      return identities[max_idx]
       # # print("personal active 2")
       #  return ('personal', None)


class WeightAdjuster:
    def __init__(self, G, W, dima_beta, agent_features=None, normative_groups=None):
        self.G = G
        self.W = W
        self.dima_beta = dima_beta
        self.agent_features = agent_features # needed for normative proximity checks
        self.normative_groups = normative_groups or {}
   
    def adjust(self, regulars, active_identities, clusters_cache, normative_dist_threshold=0.4):
        # print("Active_ID : ", active_identities )
        W_dyn = self.W.copy()

        for i in regulars:
            id_type, id_val = active_identities[i]
    
            if id_type == 'personal':
                # print(f"[adjust] Agent {i} has personal identity. Skipping adjustment.")
                continue
    
            clusters, _, _ = clusters_cache.get(i, (None, None, None))
    
            if id_type == 'dynamic':
                group = clusters.get(id_val, []) if clusters is not None else []
                neighbors = list(self.G.neighbors(i))
                upweighted = 0
                downweighted = 0
    
                for j in neighbors:
                    if j in group:
                        W_dyn[i, j] *= (1 + self.dima_beta)
                        upweighted += 1
                    else:
                        W_dyn[i, j] *= (1 - self.dima_beta)
                        downweighted += 1
                # print('upweighted: {}, Downweighted: {}'.format(upweighted,downweighted))

                # print(f"[adjust] Agent {i} (dynamic:{id_val}) - upweighted {upweighted} in-group, downweighted {downweighted} out-group neighbors.")

            elif id_type == 'normative':
                proto = self.normative_groups.get(id_val, None)
                if proto is None or self.agent_features is None:
                    # print(f"[adjust] Agent {i} (normative:{id_val}) - missing prototype or features. Skipping.")
                    continue
    
                neighbors = list(self.G.neighbors(i))
                close_neighbors = []
                far_neighbors = []
    
                for j in neighbors:
                    dist = np.linalg.norm(self.agent_features[j] - proto)
                    if dist <= normative_dist_threshold:
                        close_neighbors.append(j)
                    else:
                        far_neighbors.append(j)
    
                for j in close_neighbors:
                    W_dyn[i, j] *= (1 + self.dima_beta)
                for j in far_neighbors:
                    W_dyn[i, j] *= (1 - self.dima_beta)


            else:
                # print(f"[adjust] Agent {i} has unknown identity type: {id_type}. Skipping.")
                continue
    
            # Normalize row i
            if W_dyn[i].sum() > 0:
                W_dyn[i] /= W_dyn[i].sum()
            # else:

           #     print(f"[adjust] Warning: W_dyn row {i} sum is zero. Skipping normalization.")
        return W_dyn




