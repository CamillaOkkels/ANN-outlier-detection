import hnswlib 
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.neighbors import NearestNeighbors
import glob
import csv
import time




datasets = glob.glob("data/Kitsune*.npz")

results_fn = "results_blackbox_kitsune.csv"

fields = ['dataset', 'M', 'ef_c', 'ef_s', 'k', 'variant', 'ROCAUC', "run", 'time_build', 'time_detect']

csvfile = open(results_fn, 'w', newline='')
reswriter = csv.DictWriter(csvfile, fieldnames=fields)
reswriter.writeheader()

def _local_reachability_density(distances_X, neighbors_indices, k):
    """The local reachability density (LRD)

    The LRD of a sample is the inverse of the average reachability
    distance of its k-nearest neighbors.

    Parameters
    ----------
    distances_X : ndarray of shape (n_queries, self.n_neighbors)
        Distances to the neighbors (in the training samples `self._fit_X`)
        of each query point to compute the LRD.

    neighbors_indices : ndarray of shape (n_queries, self.n_neighbors)
        Neighbors indices (of each query point) among training samples
        self._fit_X.

    Returns
    -------
    local_reachability_density : ndarray of shape (n_queries,)
        The local reachability density of each sample.
    """
    dist_k = distances_X[neighbors_indices, k - 1]
    reach_dist_array = np.maximum(distances_X, dist_k)

    # 1e-10 to avoid `nan' when nb of duplicates > n_neighbors_:
    return 1.0 / (np.mean(reach_dist_array, axis=1) + 1e-10)
    #return 1.0 / (np.mean(reach_dist_array, axis=1) + 1e-10)
    
print(datasets)
for d in datasets:
    print(f'running on {d}')

    data = np.load(d)
    X, y = data['X'], data['y']
    if y.shape[1] > 1:
        y = y[:, 1]
    if "Mirai" in d:
        y = 1 - y
    
    for run in range(1):
        for ef_c in [200]:
            for M in [8, 16, 32, 48]:
                print(f"ef_construction: {ef_c}, M: {M}")
                start_build = time.time()
                index = hnswlib.Index(space='l2', dim=X.shape[1])
                index.init_index(max_elements=X.shape[0], ef_construction=ef_c, M = M, random_seed = 42)
                index.add_items(X, np.arange(X.shape[0]))
                end_build = time.time()
                for k in [15, 30, 50, 100]:
                    for ef_search in [8, 16, 32, 48, 64, 128, 256]:
                        start_detect = time.time()
                        index.set_ef(ef_search)
     
                        I_HNSW, D_HNSW = index.knn_query(X, k + 1)
                        D_HNSW = D_HNSW[:, 1:]
                        I_HNSW = I_HNSW[:, 1:]
                        HNSW_lrd = _local_reachability_density(
                            D_HNSW,
                            I_HNSW,
                            k
                        )

                        lrd_ratios_array = HNSW_lrd[I_HNSW] / HNSW_lrd[:, np.newaxis]

                        scores = np.mean(lrd_ratios_array, axis=1)
                        end_detect = time.time()


                        data = {
                            'dataset': d.split("/")[-1],
                            'M': M,
                            'ef_c': ef_c,
                            'ef_s': ef_search,
                            'k': k,
                            "run": run,
                            "variant": "LOF",
                            'ROCAUC': roc_auc_score(y, scores),
                            'time_build': end_build - start_build,
                            'time_detect': end_detect - start_detect,
                        }

                        print(f"dataset={d.split('/')[-1]}, k={k}, ef_search={ef_search}, LOF: {roc_auc_score(y, scores)}")
                        
                        reswriter.writerow(data)

                        # use distance to k-NN as classifier
                        scores = np.sum(D_HNSW, axis=1)
                        data = {
                            'dataset': d.split("/")[-1],
                            'M': M,
                            'ef_c': ef_c,
                            'ef_s': ef_search,
                            'k': k,
                            "run": run,
                            "variant": "KNN",
                            'ROCAUC': roc_auc_score(y, scores),
                            'time_build': end_build - start_build,
                            'time_detect': end_detect - start_detect,
                        }

                        print(f"dataset={d.split('/')[-1]}, k={k}, ef_search={ef_search}, KNN: {roc_auc_score(y, scores)}")
                        
                        reswriter.writerow(data)

csvfile.close()







