import hnswlib 
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import glob

datasets = [
    '1_ALOI.npz', 
    # '9_census.npz', 
    '10_cover.npz', 
    '17_InternetAds.npz'
]

datasets = glob.glob("*.npz")
datasets.remove('9_census.npz')



for d in datasets:

    print(f'running on {d}')

    data = np.load(d)
    X, y = data['X'], data['y']

    for k in [10]: # [10, 20, 50, 100]:

        y_hat_LOF = LOF(n_neighbors=k).fit(X).decision_scores_
        print(f"Baseline (LOF, k={k}):", roc_auc_score(y, y_hat_LOF))

        y_hat_KNN = KNN(n_neighbors=k).fit(X).decision_scores_
        print(f"Baseline (KNN, k={k}):", roc_auc_score(y, y_hat_KNN))
    
    ef_C = [10, 50, 100, 200]
    M = [4, 8, 10, 20, 40]

    ROC_contrastF = np.zeros([len(M), len(ef_C)])
    ROC_contrastT = np.zeros([len(M), len(ef_C)])

    for ef_c in range(len(ef_C)):
        for m in range(len(M)):
            try:
                print(f"ef_construction: {ef_C[ef_c]}, M: {M[m]}")
                index = hnswlib.Index(space='l2', dim=X.shape[1])
                index.init_index(max_elements=X.shape[0], ef_construction=ef_C[ef_c], M = M[m])
                index.add_items(X, np.arange(X.shape[0]))

                scores_contrastF = index.detect_outliers(contrast=False)
                
                roc_auc_contrastF = roc_auc_score(y, scores_contrastF)
                print("HNSW (distances):", roc_auc_contrastF)
                ROC_contrastF[m, ef_c] = roc_auc_contrastF - roc_auc_score(y, y_hat_KNN)


                scores_contrastT = index.detect_outliers(contrast=True)

                roc_auc_contrastT = roc_auc_score(y, scores_contrastT)
                print("HNSW (contrast):", roc_auc_contrastT)
                ROC_contrastT[m, ef_c] = roc_auc_contrastT - roc_auc_score(y, y_hat_LOF)
            except:
                pass

    fig, axes = plt.subplots(
                            nrows=1,
                            ncols=2,
                            sharex=True,
                            sharey=True,
                            figsize=(12, 15))
    
    im = axes.flat[0].imshow( ROC_contrastF, 
                             vmin = -.25, 
                             vmax = .25 )
    im = axes.flat[1].imshow( ROC_contrastT, 
                             vmin = -.25, 
                             vmax = .25 )

    xlabs = ef_C
    ylabs = M

    axes.flat[0].set_title('Contrast: false')
    axes.flat[1].set_title('Contrast: true')

    axes.flat[1].set_xticks(np.arange(len(xlabs)))
    axes.flat[1].set_xticklabels(xlabs)
    axes.flat[1].set_yticks(np.arange(len(ylabs)))
    axes.flat[1].set_yticklabels(ylabs)


    fig.text(0.5, 0.04, 'ef_C', ha='center')
    fig.text(0.04, 0.5, 'M', va='center', rotation='vertical')
    fig.text(0.2, 0.7, f'LOF: {roc_auc_score(y, y_hat_LOF):.2f}, KNN: {roc_auc_score(y, y_hat_KNN):.2f}')
    # fig.supxlabel('ef_C')
    # fig.supylabel('M')

    cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, 
             cax=cax, 
             **kw)
    
    plt.show()
    plt.savefig(f"heatmap-{d}.png")

    
    
        








