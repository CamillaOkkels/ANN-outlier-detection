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

results_fn = "results_whitebox_with_repetitions.csv"

fields = ['dataset', 'M', 'ef', 'variant', 'run', 'ROCAUC', 'time_build', 'time_detect']

csvfile = open(results_fn, 'w', newline='')
reswriter = csv.DictWriter(csvfile, fieldnames=fields)
reswriter.writeheader()
    
for d in datasets:

    print(f'running on {d}')

    data = np.load(d)
    X, y = data['X'], data['y']
    for run in range(10):
        for ef_c in [10, 50, 100, 200]:
            for M in [8, 16, 32, 48]:
                print(f"ef_construction: {ef_c}, M: {M}")
                start_build = time.time()
                index = hnswlib.Index(space='l2', dim=X.shape[1])
                index.init_index(max_elements=X.shape[0], ef_construction=ef_c, M = M)
                index.add_items(X, np.arange(X.shape[0]))
                end_build = time.time()

                scores = index.detect_outliers(contrast=False)
                end_detect_1 = time.time()

                #print("HNSW (distances):", roc_auc_score(y, scores))

    # fields = ['dataset', 'M', 'ef', 'variant', 'ROCAUC', 'time_build', 'time_detect']



                scores_contrast = index.detect_outliers(contrast=True)
                end_detect_2 = time.time()
                scores_contrast = np.nan_to_num(scores_contrast)

                data_no_contrast = {
                    'dataset': d.split("/")[-1],
                    'M': M,
                    'ef': ef_c,
                    "run": run,
                    'variant': 'contrast=False',
                    'ROCAUC': roc_auc_score(y, scores),
                    'time_build': end_build - start_build,
                    'time_detect': end_detect_1 - end_build,
                }

                data_contrast = {
                    'dataset': d.split("/")[-1],
                    'M': M,
                    'ef': ef_c,
                    'variant': 'contrast=True',
                    'run': run,
                    'ROCAUC': roc_auc_score(y, scores_contrast),
                    'time_build': end_build - start_build,
                    'time_detect': end_detect_2 - end_detect_1,
                }

                #print("HNSW (contrast):", roc_auc_score(y, scores))

                reswriter.writerow(data_no_contrast)
                reswriter.writerow(data_contrast)

csvfile.close()







