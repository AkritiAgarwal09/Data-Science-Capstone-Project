# UCI Bank Marketing modeling

This directory compares binary classifiers for predicting whether a customer will subscribe to a term deposit using the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing). Call duration is excluded because it is unavailable before a call.

- `uci_baseline.ipynb` compares eight classifiers with stratified cross-validation, selects by validation PR-AUC, and evaluates the selected model on a held-out test split.
- `uci_smotenc.ipynb` repeats the comparison with categorical-aware SMOTENC oversampling of the training class. Set `N_SYNTHETIC` in its first code cell to choose the number of synthetic subscriber rows (currently 5,000). Validation and test data remain at their original class balance.
- `scripts/uci_preprocessing.py` fetches, splits, imputes, encodes, and scales the data. `scripts/uci_preprocessing_smotenc.py` adds the configurable training-only SMOTENC step.

Both notebooks use the same default 60/20/20 train/validation/test split and show metric tables, graphs, and short interpretations. Run them from this directory using the `bead` Python environment with `ucimlrepo`, scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost, pandas, NumPy, and Matplotlib installed. The UCI fetch requires internet access.

SMOTENC handles categorical values during sampling; one-hot encoding happens afterward. Sampling and preprocessing are fitted separately inside each training fold.

## Metrics

The positive class is a customer who subscribes (`y = yes`). Precision, recall, F1, balanced accuracy, and accuracy use a predicted-probability threshold of 0.5. Higher values are better for every metric below.

| Metric | Meaning |
| --- | --- |
| PR-AUC | Area under the precision–recall curve across probability cutoffs, calculated by trapezoidal integration. It is the model-selection metric; the positive-class rate is about 0.117 here. |
| Precision | Of customers predicted to subscribe, the fraction who actually subscribe: `TP / (TP + FP)`. |
| Recall | Of actual subscribers, the fraction correctly identified: `TP / (TP + FN)`. |
| F1 | Harmonic mean of precision and recall: `2 × precision × recall / (precision + recall)`. |
| Accuracy | Fraction of all predictions that are correct. It can look high even when many subscribers are missed. |

`TP`, `FP`, and `FN` mean true positives, false positives, and false negatives. Cross-validation tables show the mean and standard deviation (`SD`) across three stratified folds. Validation metrics select the model; test metrics report its final held-out performance.

