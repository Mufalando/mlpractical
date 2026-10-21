# Coursework 1 — S2827027

`Coursework_1.ipynb` contains the implementation; the report presents Table 3, Figure 4 and draft answers to Q1, Q2 and Q4. Table 4, Figure 5 and final test evaluation are pending CW2 work.

## Pre-requisite Information:
### Common regularisation setting and models training:
1. Three hidden layers of 128 ReLU units, 47 outputs
2. Xavier uniform weights
3. Zero biases
4. Adam optimizer with learning rate `1e-4`
5. Models trained with batch size 100, 100 epochs and seed 111020 reset per run.

### Some non-standard calculation used in this repo:
1. L1/L2 penalties include weights and biases, L2 divide coefficient by 2.
2. Generalization gap formulated by subtracting validation error to training error instead of the other way around.

### Folder explanation:
1. notebooks/: Folder containing jupyter notebooks, include created model stored in cw1_results/
2. mlp/: Folder used to store MLP course's python library
3. report/: Folder used to store reports, including latex and figures
4. data/: Folder containing used dataset

## Reproduction

1. Use Python 3.12 with Jupyter, PyTorch, NumPy and Matplotlib with Intel XPU (for personal use) support.
2. Place the supplied `emnist-train.npz` and `emnist-valid.npz` in `data/`, with the notebook in `notebooks/`.
3. Open it from `notebooks/`, run imports and helper definitions, then the regularisation settings and relevant experiment cells. Skip the supplied baseline/width/depth runs. Set `load_prev_result=True` in runner calls to load saved results instead of retraining.

## Result files

Each filename begins with `depth3_width128_` and follow these suffixes rules:

| Filename suffix | Configuration |
|---|---|
| `dropout_keep{q}.npz` | Dropout keep probability `q`; required run: 0.7 |
| `L1{coeff}.npz` | L1 coefficient; required run: 0.001 |
| `L2{coeff}.npz` | L2 coefficient; required run: 0.001 |
| `label_smoothing0.1.npz` | Label smoothing alpha = 0.1 |
| `naive_elastic_net{l1}{l2}.npz` | Elastic-net coefficients, concatenated in filename |

Each `.npz` contains `stats`, `gradients`, `parameter_names` and JSON `config`. `stats` contains model's training evaluation from baseline to epochs 100. Each `stats` contain values that represent training CE, validation CE, training accuracy and validation accuracy in order. Accuracy is stored as a fraction. Reported errors use original-label cross-entropy in evaluation mode, excluding penalties and label smoothing.

Datasets and saved model weights are not included in the submission.
