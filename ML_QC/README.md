## Quality Cuts with Machine Learning: AP-SVM

These machine learning-based quality cuts are based on the strategy developed by Esteban Leon: [AP-SVM-Data-Cleaning](https://github.com/esleon97/AP-SVM-Data-Cleaning/tree/main/train)

We use a two-step approach combining **Affinity Propagation (AP)** and **Support Vector Machine (SVM)** to classify waveform quality.

---

#### 1. Affinity Propagation (AP)

AP is an unsupervised clustering algorithm used to group similar waveforms. The goal of this step is to identify clusters corresponding to different waveform types—one of which represents the "normal" or "good" waveforms we want to retain.

Due to its high memory usage, AP is applied only to a subset of the data (typically ≤ 10,000 waveforms).

We tune the AP hyperparameters (`preference` and `damping`) so that it yields around 100 clusters. These clusters are initially labeled with arbitrary numeric IDs (e.g., 1 to *n_clusters*). In a **manual** post-processing step, we map these numeric labels to meaningful quality control (QC) labels using a predefined legend.

| QC Label | Description   |
| -------- | ------------- |
| 0        | normal        |
| 1        | neg. going    |
| 2        | up slope      |
| 3        | down slope    |
| 4        | spike         |
| 5        | x-talk        |
| 6        | slope rising  |
| 7        | early trigger |
| 8        | late trigger  |
| 9        | saturated     |
| 10       | soft pileup   |
| 11       | hard pileup   |
| 12       | bump          |
| 13       | noise         |


#### 2. Support Vector Machine (SVM)

SVM is a supervised learning algorithm. We train the SVM on the same waveforms used for AP, using the QC labels assigned in the previous step.

Once trained, the SVM model can predict QC labels for new, unseen waveforms—effectively scaling the AP-based classification to larger datasets.

### Overview of AP-SVM Workflow Scripts

The following scripts implement the AP-SVM quality cut workflow:

- **`AP_hyperpars_opt.jl`**  
  Runs the Affinity Propagation (AP) model over a grid of hyperparameter combinations to find best combination.  
  You have to manually select a set of parameters that results in approximately `100` clusters for use in the next step. Can be run with SLURM script: 
  
  `sbatch run_AP_opt.sh` on a computing cluster (might need modifications for your cluster specs).  

- **`01_AP_train.jl`**  
  Runs the AP clustering algorithm and saves intermediate results, including:
  - AP labels for each waveform
  - Indices of cluster centers (waveforms)

- **`02_AP_train.jl`**  
  Performs **manual** relabeling of cluster centers, mapping AP labels to meaningful QC labels.  
  Saves the updated AP results.

- **`03_SVM.jl`**  
  Trains and evaluates the Support Vector Machine (SVM) model.  
  The waveforms used in AP are split into training and test sets.  
  The script validates the classification performance of the SVM. Save SVM results.

- **`04_Apply_AP-SVM.jl`**  
  Applies the trained AP-SVM model to unseen data, such as all waveforms from a full run.

> **Note:** SVM hyperparameters are currently not optimized.  
> Future improvements should include hyperparameter tuning to enhance model performance.
