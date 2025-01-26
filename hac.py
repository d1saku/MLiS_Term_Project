import numpy as np
import scipy as sp


class HAC():
    """
    This is an improved interation upon naive implementation.
    will document later :D
    """
    def __init__(self, n_clusters, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X):
        n_clusters = X.shape[0]
        clusters = {i: [i] for i in range(n_clusters)}
        active_clusters = list(clusters.keys())
        distance_matrix = self._distance_matrix(X)

        while len(active_clusters) > self.n_clusters:
            min_distance = np.inf
            clusters_to_merge = (0, 0)
            
            for i in range(0, len(active_clusters), 1):
                for j in range(i+1, len(active_clusters), 1):
                    cid_i = active_clusters[i]
                    cid_j = active_clusters[j]
                    dist_ij = distance_matrix[cid_i, cid_j]
                    if dist_ij < min_distance:
                        min_distance = dist_ij
                        clusters_to_merge = (cid_i, cid_j)
                    # for datapoints_i in clusters[cid_i]:
                    #     for datapoints_j in clusters[cid_j]:
                    #         if (distance_matrix[datapoints_i, datapoints_j] < min_distance):
                    #             min_distance = distance_matrix[i, j]
                    #             clusters_to_merge = (cid_i, cid_j)

            cluster1, cluster2 = clusters_to_merge
            clusters[cluster1].extend(clusters[cluster2])

            for cluster in active_clusters:
                if cluster == cluster1:
                    continue

                if self.linkage == 'single':
                    new_dist = min(distance_matrix[cluster1, cluster],
                                   distance_matrix[cluster2, cluster])
                elif self.linkage == 'complete':
                    new_dist = max(distance_matrix[cluster1, cluster],
                                   distance_matrix[cluster2, cluster])
                else:
                    raise ValueError("invalid linkage type passed as an argument")
                
                distance_matrix[cluster1, cluster] = new_dist
                distance_matrix[cluster, cluster1] = new_dist

            del clusters[cluster2]
            active_clusters.remove(cluster2)

            self.labels_ = np.zeros(n_clusters, dtype=int)
            label_id = 0
            for cid in clusters:
                for idx in clusters[cid]:
                    self.labels_[idx] = label_id
                label_id += 1

        return self


    def _distance_matrix(self, X):
        n_clusters = X.shape[0]
        distance_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                distance_matrix[i][j] = sp.spatial.distance.euclidean(X[i], X[j])

        return distance_matrix
