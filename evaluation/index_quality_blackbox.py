import hnswlib
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.neighbors import NearestNeighbors

import csv
import glob

datasets = [
#    '1_ALOI.npz', 
#   '9_census.npz', 
    'data/10_cover.npz', 
#    '17_InternetAds.npz'
]

datasets = glob.glob("data/*.npz")
datasets = [x for x in datasets if "Kitsune" not in x]

interesting_datasets = []
for d in datasets: 
    data = np.load(d)
    X, y = data['X'], data['y']
    interesting_datasets.append((d, X.shape))

interesting_datasets = [fn[0] for fn in sorted(interesting_datasets, key=lambda x: x[1][0] * x[1][1])[-15:]]
print(interesting_datasets)
datasets = interesting_datasets


results_fn = "results_recall_blackbox.csv"

hnsw_fields = ['dataset', 'M', 'ef_c', 'ef_s', 'k', 'recall.25', 'recall', 'recall.75']

csvfile_hnsw = open(results_fn, 'w', newline='')
reswriter2 = csv.DictWriter(csvfile_hnsw, hnsw_fields)
reswriter2.writeheader()

for d in datasets:
    print(f"Running on {d}")

    data = np.load(d)
    X, y = data['X'], data['y']
    D, I = NearestNeighbors(n_neighbors=100).fit(X).kneighbors(n_neighbors=100)

    for k in [15, 30, 50, 100]:

        if len(X) < k + 1:
            print(f"Skipping {d}, only {len(X)} samples.")
            continue


        for ef_c in [10, 50, 100, 200]:
            for M in [8, 16, 32, 48]:
                index = hnswlib.Index(space='l2', dim=X.shape[1])
                index.init_index(max_elements=X.shape[0], ef_construction=ef_c, M = M)
                index.add_items(X, np.arange(X.shape[0]))

                for ef_search in [8, 16, 32, 48, 64, 128]:
                    index.set_ef(ef_search)
                    # sometimes we don't get enough candidates returned for small values which will break the code below.
                    try:
                        I_HNSW, D_HNSW = index.knn_query(X, k + 1)
                        I_HNSW = I_HNSW[:, 1:]
                        recalls = np.zeros(len(X))
                        for v, indices in enumerate(I_HNSW):
                            recalls[v] = len(set(indices).intersection(set(I[v][:k]))) / k
                        print(np.mean(recalls))

                        data = {
                            "dataset": d.split("/")[1],
                            "M": M,
                            "ef_c": ef_c,
                            "ef_s": ef_search,
                            "k": k,
                            'recall.25': np.percentile(recalls, 25), 
                            'recall': np.mean(recalls), 
                            'recall.75': np.percentile(recalls, 75),
                        }
                        reswriter2.writerow(data)
                    except:
                        pass





