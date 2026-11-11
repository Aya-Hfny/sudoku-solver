# Data Dictionary — P7 Handwritten Sudoku Reader and Solver

## A. Real MNIST source

MNIST source images are grayscale 28×28 handwritten digits with integer labels 0–9. The pipeline downloads the official train/test splits using `torchvision.datasets.MNIST`. The source contains 60,000 training records and 10,000 test records.

## B. Preprocessed `mnist_1_9.npz`

| Field | Type / shape | Source | Description |
|---|---|---|---|
| `x_train` | float32 `[N_train,28,28]` | Real MNIST train | Pixel intensity divided by 255; values in [0,1] |
| `y_train` | int64 `[N_train]` | Real MNIST train labels | Target digit 1–9; class 0 removed |
| `x_val` | float32 `[N_val,28,28]` | Real MNIST train | Validation partition, same normalization |
| `y_val` | int64 `[N_val]` | Real MNIST train labels | Target digit 1–9; class 0 removed |
| `x_test` | float32 `[N_test,28,28]` | Official MNIST test | Held-out test images, normalized; class 0 removed |
| `y_test` | int64 `[N_test]` | Official MNIST test labels | Target digit 1–9 |

`N_train` and `N_val` are below 50,000 and 10,000 because the raw 60,000 records are split first, then zero labels are dropped. `N_test` is below 10,000 for the same class-filtering reason. The official test records are never used for fitting or temperature calibration.

## C. `data_dictionary.json`

| Field | Type | Meaning |
|---|---|---|
| `seed` | integer | Seed used for the split |
| `source` | string | `MNIST` |
| `filter` | string | Class 0 excluded; classes 1–9 retained |
| `normalization` | string | float32 pixels divided by 255 |
| `split` | string | Exact raw train/validation split and filtering description |
| `arrays` | object | Shape and dtype for each saved array |

## D. Generated `puzzles.jsonl`

Each JSONL line describes one generated Sudoku sample.

| Field | Type | Source | Meaning |
|---|---|---|---|
| `seed` | integer | Generator | Unique reproducibility seed for this board |
| `solution` | 9×9 integer list | Generated | Complete valid Sudoku answer (ground truth) |
| `puzzle` | 9×9 integer list | Generated | Clue grid after removing cells while preserving uniqueness; 0 = blank |
| `givens` | integer | Derived | Number of nonzero cells in `puzzle` |
| `mosaic_path` | string | Generated | Relative filename of 252×252 image mosaic |
| `injected_misreads` | list of objects | Generated | Injected recognition errors, each with `row`, `col`, `true_digit`, `read_digit` |
| `read_grid` | 9×9 integer list | Derived | `puzzle` plus injected wrong readings |
| `hard_digit_rule` | string | Generated | Documents the hard-cell synthesis: true-class MNIST glyph blended with wrong-class distractor at alpha 0.45 |

## E. Generated mosaic images

| Property | Value | Meaning |
|---|---|---|
| Dimensions | 252×252 pixels | 9×9 cells, each 28×28 |
| Color mode | 8-bit grayscale | MNIST-compatible black background / white ink |
| Digit source | Real MNIST training images | A random MNIST example matching the intended true digit |
| Injected hard cells | Generated image mixtures | True-class glyph blended with a wrong-class distractor at alpha 0.45; metadata retains true/read digits |
| Blank cells | Black pixels, no digit | Blanks are detected by a brightness/ink threshold, not by the digit classifier |
| Grid | Synthetic lines | Known 9×9 grid and 3×3 box boundaries |

## F. Model artifacts

| File / field | Type | Meaning |
|---|---|---|
| `digit_cnn.pt` | PyTorch state dict | Trained CNN weights for nine classes |
| `calibration.json.temperature` | float | Positive temperature fitted on validation logits |
| `cnn_metrics.json` | JSON object | CNN validation/test accuracy, ECE, NLL and confusion matrices |
| `all_metrics.json.logistic_regression_test` | JSON object | Logistic Regression test metrics |

## G. Experiment CSV `experiment_results.csv`

| Column | Type | Meaning |
|---|---|---|
| `experiment` | integer | Experiment number 1, 2 or 3 |
| `seed` | integer | Puzzle/experiment seed |
| `setting` | integer | Misread count for experiment 1/3; givens count for experiment 2 |
| `method` | string | Search/CSP/correction method |
| `expanded` | integer | Search/CSP nodes expanded |
| `generated` | integer | Search states or CSP assignments generated |
| `recovery` | integer 0/1 | 1 iff the final completed Sudoku exactly equals the generated ground-truth solution |

## H. Correction log

Each correction contains `row`, `col`, `old`, `new`, and `cost`. Rows and columns are zero-indexed. `cost` is the nonnegative change cost computed from the probability distribution for that cell.
