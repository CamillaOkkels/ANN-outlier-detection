# HNSW with outlier detection methods

This describes the experiments carried out in the paper 

> On the Design of Scalable Outlier Detection Methods Using Approximate Nearest Neighbor Graphs, Camilla Birch Okkels, Martin Aumüller & Arthur Zimek, SISAP 2024. 

For the experiments to run, the modified version of hnswlib <../hnswlib> needs to be installed on the system. Furthermore, baselines are running using [PyOD](https://github.com/yzhao062/pyod).
Please run `pip install -r requirements.txt` to install these dependencies. 

## Data preparation

The experiments were run on two sets of data, one found from the AdBench repository and the other from the Kitsune library.

The AdBench datasets can be found here: https://github.com/Minqi824/ADBench/tree/main 
under adbench/datasets/Classical/

The Kitsune datasets can be found and downloaded from here: https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset

## Experiments

You can carry out all experiments in the paper by running `bash experiments.sh`.

**what does the user have to do after running the experiments? move some csv files around**?

This will create the individual csv files that are used in the analysis. The evaluation is present in the jupyter notebook `evaluation/`.
The jupyter notebook 'evaluation/' also creates and reads the file critddiagram.csv from which the critddiagram.tex file is created - this file contains code to create the critical difference plot in a LaTeX document.

Note when the file critddiagram is first created it contains more than the 12 datasets covered in the paper. The datasets not containing all 7 methods (3 baseline, 2 blackbox and 2 whitebox) have to be removed manually before the .tex file can be created.
