import hnswlib 
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.neighbors import NearestNeighbors
import glob
import csv
import time



datasets = [
#    '1_ALOI.npz', 
#   '9_census.npz', 
    '10_cover.npz', 
#    '17_InternetAds.npz'
]

datasets = glob.glob("data/*.npz")

results_fn = "results_baselines.csv"

fields = ['dataset', 'variant',  'k', 'ROCAUC', 'time']

csvfile = open(results_fn, 'w', newline='')
reswriter = csv.DictWriter(csvfile, fieldnames=fields)
reswriter.writeheader()


def ODIN(X, k):
    k += 1
    _, I = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(n_neighbors=k)
    scores = np.ones(len(X))
    for i in range(len(X)):
        for j in range(k):
            scores[I[i,j]] += 1
    scores = 1 / scores
    return scores

# def KNNGraph(X, k):
#     k += 1
#     D, I = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(n_neighbors=k)
#     d = np.zeros(len(X))
#     for i in range(len(X)):
#         d[i] += np.sum(D[i])
#     scores = np.zeros(len(X))
#     for i in range(len(X)):
#         s = sum([d[j] for j in I[i]])
#         scores[i] = s / d[i]
#     return scores
    
methods = { 
    "LOF": lambda X, k: LOF(n_neighbors=k).fit(X).decision_scores_,
    "KNN": lambda X, k: KNN(n_neighbors=k).fit(X).decision_scores_,
    "ODIN": lambda X, k: ODIN(X,k),
}
for d in datasets:

    print(f'running on {d}')

    data = np.load(d)
    X, y = data['X'], data['y']

    for k in [15, 30, 50, 100]:
        if k > X.shape[0]:
            continue

        for method, foo in methods.items():
            start = time.time()
            y_hat = foo(X, k)
            end = time.time()


            print(f"Baseline ({method}, k={k}):", roc_auc_score(y, y_hat))

            data = {
                'dataset': d.split("/")[1],
                'variant': method,
                'k': k,
                'ROCAUC': roc_auc_score(y, y_hat),
                'time': end - start,
            }

            reswriter.writerow(data)

csvfile.close()