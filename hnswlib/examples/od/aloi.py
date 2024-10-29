import hnswlib 
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from sklearn.metrics import roc_auc_score
import numpy as np
import time

datasets = [
#    '1_ALOI.npz', 
#    '9_census.npz', 
#    '10_cover.npz', 
    '17_InternetAds.npz'
]

for d in datasets:

    print(f'running on {d}')

    data = np.load(d)
    X, y = data['X'], data['y']

    for k in [10, 20, 50, 100]:

        y_hat = LOF(n_neighbors=k).fit(X).decision_scores_
        print(f"Baseline (LOF, k={k}):", roc_auc_score(y, y_hat))

        y_hat = KNN(n_neighbors=k).fit(X).decision_scores_
        print(f"Baseline (KNN, k={k}):", roc_auc_score(y, y_hat))


    for ef_c in [10, 50, 100, 200]:
        for M in [4, 8, 10, 20, 40]:
            print(f"ef_construction: {ef_c}, M: {M}")
            start = time.time()
            index = hnswlib.Index(space='l2', dim=X.shape[1])
            index.init_index(max_elements=X.shape[0], ef_construction=ef_c, M = M)
            index.add_items(X, np.arange(X.shape[0]))
            end = time.time()

            #print(f"Building the index took {end - start:.2f} s.")

            # index.set_ef(100)

            # start = time.time()

            # index.knn_query(X, k=15)

            # print(f"Search the index took {time.time() - start:.2f} s.")


            scores = index.detect_outliers(contrast=False)

            print("HNSW (distances):", roc_auc_score(y, scores))

            # start = time.time()

            scores = index.detect_outliers(contrast=True)
            # print(f"Detecting outliers directly took {time.time() - start:.2f} s.")

            print("HNSW (contrast):", roc_auc_score(y, scores))







