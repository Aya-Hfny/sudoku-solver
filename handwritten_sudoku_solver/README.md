# P7 — Handwritten Sudoku Reader and Solver

A reproducible prototype that recognizes handwritten digits in a **synthetic 9×9 MNIST mosaic**, detects conflicting readings, proposes low-cost digit corrections, and solves the corrected Sudoku with a constraint-satisfaction problem (CSP).

> **Scope / honesty note:** MNIST contains isolated handwritten digits, not photographed Sudoku sheets. This project creates Sudoku puzzles and mosaics from real MNIST digit crops and assumes the 9×9 grid boundaries are known. It does not claim camera capture, perspective correction, or automatic grid localization. The normal demo uses explicitly labelled controlled probabilities until a CNN has been trained; `--mosaic` mode uses real CNN inference once weights exist.

## 1. What the project implements

- **Data:** reproducible MNIST download and preprocessing, digits 1–9 only, normalization to `[0,1]`, seeded 50k/10k split of the original 60k training records, untouched official test partition apart from removing class 0.
- **NN:** compact CNN, calibrated probabilities using validation-only temperature scaling, accuracy, confusion matrix, ECE and NLL; Logistic Regression comparison on a deterministic 20,000-image training subset.
- **Puzzle generator:** randomized backtracking to create complete valid boards; clue removal while preserving uniqueness; seeded 9×9 MNIST mosaics; conflicting injected misreads with cell-level ground truth; hard cells use a documented true-glyph / wrong-class-distractor blend.
- **Search:** UCS, Greedy Best-First and A* over correction states; nonnegative likelihood-ratio cost; top-3 alternative classes; configurable correction budget; generated/expanded counters.
- **CSP:** 81 variables, row/column/box inequality constraints, MRV variable selection, Backtracking, Forward Checking, AC-3, optional uniqueness check, and assignment/backtrack/pruning/node counters.
- **Integration:** NN probabilities → AC-3 consistency check → correction Search → CSP solvability/optional uniqueness check → correction log. Unrecoverable inputs return `unreadable` and least-confident cells.
- **Evaluation:** the three required experiments, CSV, mean/population-SD summary and plots. Experiment 1 defaults to **100 generated puzzles at each misread level (0–4)**; Experiment 2 uses at least five seeds.

## 2. Environment and installation

Use Python 3.11 (recommended) or Python 3.10. Exact package versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the pinned PyTorch wheel is unavailable for your operating system, install a compatible PyTorch/torchvision pair from the official PyTorch installation selector, then install the remaining requirements.

## 3. Exact commands (from repository root)

```bash
# A. Download MNIST, preprocess, and write the data dictionary
python data/make_data.py --out data/processed --seed 42

# B. Train CNN, calibrate on validation data, evaluate on test data,
#    and compare with Logistic Regression
python nn/train.py --data data/processed/mnist_1_9.npz --epochs 8 --seed 42

# C. Generate 100 reproducible puzzles and 9x9 MNIST mosaics
python data/generate_puzzles.py --out data/processed/puzzles --seed 42 --count 100 --givens 35 --misreads 2

# D1. Demonstrate full correction pipeline with transparent synthetic probabilities
python run_demo.py --method ucs --max-corrections 3

# D2. Run real CNN inference on one generated mosaic (after B and C)
python run_demo.py --mosaic data/processed/puzzles/mosaic_0000.png --weights nn/artifacts/digit_cnn.pt

# E. Run the tests
pytest -q
# equivalent: python -m pytest -q

# F. Run all three experiments. At least five seeds are required.
# Experiment 1 and 3: 100 puzzles per level by default.
python eval/run_experiments.py --seeds 5 --puzzles-per-level 100 --max-expansions 1000 --out results
```

### Output files

- `data/processed/mnist_1_9.npz`, `data_dictionary.json`
- `data/processed/puzzles/puzzles.jsonl`, `mosaic_XXXX.png`
- `nn/artifacts/digit_cnn.pt`, `calibration.json`, `cnn_metrics.json`, `all_metrics.json`
- `results/experiment_results.csv`, `correction_logs.jsonl`, `summary.json`, `experiment_*.png`, `demo_pipeline.png`

The archive includes `results/pilot_5_per_level/` as a clearly labelled five-puzzle-per-level pilot. It is not the required 100-puzzle final evaluation. Run the full command above to produce the 100-puzzle-per-level final tables. The dataset and trained weights are generated locally and are not bundled into the source archive. The submission artifacts also include `REPORT.pdf` (5 pages) and `PRESENTATION.pptx` (10-minute talk plus 5-minute demo plan).

## 4. Project structure

```text
data/        make_data.py, sudoku_generator.py, generate_puzzles.py, DATA_DICTIONARY.md
nn/          model.py, train.py, inference.py, metrics.py
search/      correction.py
csp/         sudoku.py
integration/ pipeline.py
eval/        run_experiments.py
tests/       test_sudoku.py, test_search.py
results/     reproducible demo and experiment outputs
README.md    setup and exact commands
REPORT.md    editable report source
REPORT.pdf    rendered 5-page report (within 12-page limit)
PRESENTATION.md  10-minute presentation + 5-minute live demo plan
PRESENTATION.pptx 12-slide presentation deck
ENVIRONMENT_VALIDATION.md verified commands and remaining external runs
CONTRIBUTIONS_TEMPLATE.md
```

## 5. Data dictionary

See [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) for all source, generated, and derived fields. Key arrays in `mnist_1_9.npz`:

| Field | Shape / type | Meaning |
|---|---|---|
| `x_train` | `[N,28,28] float32` | Training images, scaled to 0–1 |
| `y_train` | `[N] int64` | Labels 1–9; zero removed |
| `x_val` | `[N,28,28] float32` | Validation images from original MNIST training split |
| `y_val` | `[N] int64` | Validation labels 1–9 |
| `x_test` | `[N,28,28] float32` | Official test images, zero removed; never used for fitting |
| `y_test` | `[N] int64` | Test labels 1–9 |

The seeded split is made on the original 60,000 MNIST training records (50,000 train / 10,000 validation), then class 0 is removed separately from each partition. Thus the final arrays contain slightly fewer than 50,000 and 10,000 samples. This preserves the requested split and excludes zero from the nine-class classifier.

## 6. Algorithm definitions

### Neural network and calibration

The CNN maps a 28×28 grayscale image to nine logits for digits 1–9. Softmax gives class probabilities. A scalar temperature is fitted by minimizing validation negative log-likelihood; it is applied to test/inference logits. Test data is not used for model selection or calibration. Logistic Regression uses flattened pixels and a fixed-seed subset of up to 20,000 training examples.

### Search

A state is the current 81-cell reading. An action changes one currently conflicting cell to one of its top-3 alternative NN classes. For a change from digit `d` to `d'`:

`cost = max(0, log(p(d) / p(d')))`

The cost is never negative. UCS prioritizes accumulated cost `g`; Greedy prioritizes the conflict lower bound `h`; A* prioritizes `g+h`. The A* lower bound is built from a greedy matching of disjoint duplicate-conflict edges: at least one endpoint of each selected edge must change, so summing the cheapest endpoint-change cost is a lower bound. Search stops at the configured correction limit. A candidate goal must have no duplicate givens and must pass CSP solvability.

### CSP

Each of 81 cells is a variable. A given has a singleton domain; a blank has domain `{1,…,9}`. Cells sharing a row, column, or 3×3 box must differ. MRV plus degree tie-breaking selects variables. The three comparison settings are:

- `BT`: Backtracking only (`use_fc=False, use_ac3=False`)
- `BT+FC`: Backtracking + Forward Checking
- `BT+FC+AC3`: Backtracking + Forward Checking + AC-3

AC-3 revises binary inequality arcs; FC removes a newly assigned digit from all peers. Optional uniqueness mode stops after two solutions and accepts only exactly one.

## 7. Reproducibility and experiment design

All data generators and experiments accept explicit seeds. `eval/run_experiments.py` refuses fewer than five seeds. It writes raw observations to CSV and mean plus population standard deviation to JSON.

1. **Search comparison:** UCS vs Greedy vs A* at 0–4 conflicting misreads; report exact solution recovery, expanded nodes and generated nodes. Default: 100 unique generated puzzles at each error level.
2. **CSP comparison:** BT vs BT+FC vs BT+FC+AC-3 at 25, 30 and 35 givens; report nodes, assignments and exact recovery. Default: five seeds.
3. **Correction policy:** NN-probability-guided UCS vs no correction vs random correction on two injected misreads; report exact solution recovery. Default: 100 puzzles.

The included experiment probabilities are a controlled synthetic classifier model (52% on the wrong read, 45% on the known correct digit for injected errors). They test the search/correction logic, not the measured CNN. Use the trained-CNN mosaic inference path for real classifier probability analysis.

## 8. Limitations and academic integrity

- MNIST is isolated digits, not real Sudoku photographs; the grid is synthetically assembled and its cell boundaries are known.
- Injected errors are controlled and intentionally conflict with a peer clue so that correction is observable.
- The demo adapter is synthetic. Do not report its recovery rate as CNN performance. The included pilot is five puzzles per error level; it is a development check, not a replacement for the 100-puzzle-per-level final experiment.
- The CNN and Logistic Regression metrics exist only after successfully downloading MNIST and running `nn/train.py`. This preparation environment could not download MNIST (DNS/network unavailable), so no measured classifier metrics or trained weights are bundled. See `ENVIRONMENT_VALIDATION.md`. Do not invent or copy unmeasured accuracy values into the report.
- No student names or team contributions are fabricated. Complete `CONTRIBUTIONS_TEMPLATE.md` with actual work (maximum 200 words per person), and add real team commits; the included `GIT_LOG.txt` explicitly labels the initial AI-assisted snapshot rather than claiming student history. Run `git log --oneline` for the final submission.

## 9. Dataset source and citation

The original MNIST homepage is <http://yann.lecun.com/exdb/mnist/>. It describes 60,000 training images, 10,000 test images, 28×28 grayscale digits, and four gzip files. The original page could not be fetched directly in this environment, but the NIST Biometric Database Catalog independently confirms the 60,000/10,000 split and centered fixed-size images: <https://tsapps.nist.gov/BDbC/Search/Details/387>. See also the MNIST documentation at <https://www.tensorflow.org/datasets/catalog/mnist>.

MNIST is publicly accessible. The original homepage's license terms could not be directly verified in this environment; do not redistribute the raw dataset in this source archive, and verify the current terms before redistribution.

## Run the browser interface (Windows / VS Code)

The project includes a Streamlit interface in `app.py`.

1. Open this folder in VS Code and open **Terminal → New Terminal**.
2. Create and activate a virtual environment (PowerShell):

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Prepare the MNIST subset and train the CNN (the trained weights are required by the web interface):

   ```powershell
   python data/make_data.py --out data/processed --seed 42
   python nn/train.py --data data/processed/mnist_1_9.npz --epochs 8 --seed 42
   ```

4. Generate example mosaics. The first run may download MNIST if it is not already cached:

   ```powershell
   python data/generate_puzzles.py --out data/processed/puzzles --seed 42 --count 10 --givens 35 --misreads 2
   ```

5. Start the web app:

   ```powershell
   python -m streamlit run app.py
   ```

6. Open the printed local address (normally `http://localhost:8501`) in Google Chrome. Keep the terminal open while using the app; press `Ctrl+C` in the terminal to stop it.

**Input limitation:** `nn/inference.py` currently accepts only the project's 252×252 generated grayscale mosaic layout. It does not yet detect a Sudoku grid from a normal phone/camera photograph.
