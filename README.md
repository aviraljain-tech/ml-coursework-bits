# Machine Learning Coursework – BITS Pilani

Assignments from **EEE G513: Machine Learning for Electronics Engineers** (graduate-level course, BITS Pilani, K.K. Birla Goa Campus, Jan–May 2025), written in Python with scikit-learn.

| # | Project | Methods | Key result |
|---|---------|---------|------------|
| 01 | [Activity recognition from accelerometer data](01-activity-recognition-kmeans-rf) | Signal segmentation, hierarchical k-means vector quantisation, histogram features, random forest, parameter tuning | Accuracy improved from **54.2% to 64.9%** by tuning segment length and cluster numbers |
| 02 | [Regression and residual diagnostics](02-regression-diagnostics) | Linear regression, log and cube-root transforms, residual analysis, Q-Q plots, ElasticNet with grid-search cross-validation | Comparison of transformed vs. untransformed models on three datasets |

## Setup
```bash
pip install -r requirements.txt
```
Datasets are public and come from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) (links in each folder).

**Author:** Aviral Jain
