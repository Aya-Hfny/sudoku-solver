# Environment validation and remaining external run

This file records what was actually executed while preparing this submission archive.

## Verified here

- `pytest -q`: **17 passed** after adding `pytest.ini` so direct pytest invocation resolves the project modules.
- `python run_demo.py --method ucs --max-corrections 3`: completed successfully using the documented controlled synthetic probability adapter. It corrected two injected conflicting reads, solved the puzzle, wrote `results/demo_pipeline.png`, and reported Search/CSP counters.
- `results/pilot_5_per_level/`: five generated puzzles per misread level (0–4), five seeds for the CSP experiment, and the three pilot plots/CSV/summary. These are pilot results, not the required 100-puzzle-per-level final evaluation.

## Not executed here (external dependency / compute)

- `python data/make_data.py --out data/processed --seed 42` was attempted. The MNIST download failed because this execution environment could not resolve the dataset hosts (DNS/network unavailable). No local MNIST copy was present.
- Consequently, `nn/train.py` could not be run on MNIST, no trained CNN weights or measured CNN-vs-Logistic-Regression metrics are included, and real-CNN mosaic inference remains unverified in this environment.
- The full 100-puzzle-per-misread-level experiment command was not completed here. The runner is included and the small pilot ran; do not label pilot measurements as final results.
- Individual contribution statements and student-authored git history cannot be fabricated. Complete `CONTRIBUTIONS_TEMPLATE.md` with actual team members and add genuine commits.

## Required final commands

Run from the repository root in Python 3.10/3.11 with network access:

```bash
python -m pip install -r requirements.txt
python data/make_data.py --out data/processed --seed 42
python nn/train.py --data data/processed/mnist_1_9.npz --epochs 8 --seed 42
python data/generate_puzzles.py --out data/processed/puzzles --seed 42 --count 100 --givens 35 --misreads 2
python run_demo.py --mosaic data/processed/puzzles/mosaic_0000.png --weights nn/artifacts/digit_cnn.pt
python eval/run_experiments.py --seeds 5 --puzzles-per-level 100 --max-expansions 1000 --out results/final
pytest -q
```

After running, copy measured CNN and full-experiment results into `REPORT.md` and `PRESENTATION.pptx`; do not invent missing metrics.
