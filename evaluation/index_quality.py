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


results_fn = "results_recall.csv"

fields = ['dataset', 'M', 'ef', 'k', 'recall.25', 'recall', 'recall.75', 'recall@50.25', 
          'recall@50', 'recall@50.75', 'recall@100.25', 'recall@100', 'recall@100.75', 
          'recall2hop.25', 'recall2hop', 'recall2hop.75']

hnsw_fields = ['dataset', 'M', 'ef_c', 'ef_s', 'k', 'recall.25', 'recall', 'recall.75']

csvfile = open(results_fn, 'w', newline='')
reswriter = csv.DictWriter(csvfile, fieldnames=fields)
reswriter.writeheader()

csvfile_hnsw = open("results_recall_blackbox.csv", 'w', newline='')
reswriter2 = csv.DictWriter(csvfile_hnsw, hnsw_fields)
reswriter2.writeheader()

for d in datasets:
    print(f"Running on {d}")

    data = np.load(d)
    X, y = data['X'], data['y']
    k = 100

    if len(X) < k + 1:
        print(f"Skipping {d}, only {len(X)} samples.")
        continue

    D, I = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(n_neighbors=k)

    for ef_c in [10, 50, 100, 200]:
        for M in [4, 8, 10, 20, 40]:
            index = hnswlib.Index(space='l2', dim=X.shape[1])
            index.init_index(max_elements=X.shape[0], ef_construction=ef_c, M = M)
            index.add_items(X, np.arange(X.shape[0]))

            adj = index.get_adj()
            recalls = np.zeros(len(X))
            recalls50 = np.zeros(len(X))
            recalls100 = np.zeros(len(X))
            recallstwohop = np.zeros(len(X))
            distances = np.zeros(len(X))

            for v, neighbors in enumerate(adj):
                deg = len(neighbors)
                N = set([j for i in neighbors for j in adj[i] ])
                recalls[v] = len(set(neighbors).intersection(set(I[v][:deg]))) / deg
                recalls50[v] = len(set(neighbors).intersection(set(I[v][:50]))) / deg
                recalls100[v] = len(set(neighbors).intersection(set(I[v]))) / deg
                recallstwohop[v] = len(N.intersection(set(I[v][:len(N)]))) / min(len(N), k)

            data = {
                "dataset": d,
                "M": M,
                "ef": ef_c,
                "k": k,
                'recall.25': np.percentile(recalls, 25), 
                'recall': np.mean(recalls), 
                'recall.75': np.percentile(recalls, 75),
                'recall@50.25': np.percentile(recalls50, 25),
                'recall@50': np.mean(recalls50),
                'recall@50.75': np.percentile(recalls50, 75),
                'recall@100.25': np.percentile(recalls100, 25),
                'recall@100': np.mean(recalls100),
                'recall@100.75': np.percentile(recalls100, 75),
                'recall2hop.25': np.percentile(recallstwohop, 25),
                'recall2hop': np.mean(recallstwohop),
                'recall2hop.75': np.percentile(recallstwohop, 75),
            }

            reswriter.writerow(data)


            print(f"Recall (ef={ef_c}, M={M}): 25%: {np.percentile(recalls, 25):.2f}, avg: {np.mean(recalls):.2f}, 75%: {np.percentile(recalls, 75):.2f}")
            print(f"Recall@50 (ef={ef_c}, M={M}): 25%: {np.percentile(recalls50, 25):.2f}, avg: {np.mean(recalls50):.2f}, 75%: {np.percentile(recalls50, 75):.2f}")
            print(f"Recall@100 (ef={ef_c}, M={M}): 25%: {np.percentile(recalls100, 25):.2f}, avg: {np.mean(recalls100):.2f}, 75%: {np.percentile(recalls100, 75):.2f}")
            print(f"Recall2hop@100 (ef={ef_c}, M={M}): 25%: {np.percentile(recallstwohop, 25):.2f}, avg: {np.mean(recallstwohop):.2f}, 75%: {np.percentile(recallstwohop, 75):.2f}")
            print()


            for ef_search in [4, 8, 16, 32, 48, 64, 128, 256]:
                index.set_ef(ef_search)
                # sometimes we don't get enough candidates returned for small values which will break the code below.
                try:
                    I_HNSW, D_HNSW = index.knn_query(X, k + 1)
                    I_HNSW = I_HNSW[:, 1:]
                    recalls = np.zeros(len(X))
                    for v, indices in enumerate(I_HNSW):
                        recalls[v] = len(set(indices).intersection(set(I[v]))) / k

                    data = {
                        "dataset": d,
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



csvfile.close()




