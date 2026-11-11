# Final validation status (2026-10-30)

## Verified in this environment
- The four supplied MNIST IDX gzip files were loaded through torchvision's MNIST loader.
- `data/make_data.py --out data/processed --seed 42` completed successfully.
- Generated `data/processed/mnist_1_9.npz` and `data/processed/data_dictionary.json`.
- Dataset arrays: x_train (45,073, 28, 28), y_train (45,073), x_val (9,004, 28, 28), y_val (9,004), x_test (9,020, 28, 28), y_test (9,020). Pixel values are float32 in [0,1]; labels are 1..9.
- `python -m pytest -q`: 17 passed.
- `python run_demo.py --method ucs --max-corrections 3`: completed; two injected conflicts were corrected, CSP solved the board, correction log and counters were printed, and `results/demo_pipeline.png` was produced.

## Still required before claiming a fully final submission
- Train the CNN and produce measured calibrated CNN vs Logistic Regression test metrics. The training command was attempted but exceeded the execution time available in this environment; no trained weights or measured NN metrics are included.
- Run the full experiment command with at least five seeds and 100 puzzles per misread level. The archive contains only the explicitly labelled five-puzzle pilot; do not report pilot figures as final.
- Fill in each student's real individual contribution statement and ensure the included git history represents the team's actual work.
- Run real-CNN mosaic inference after training the weights.

## Commands to finish
```bash
python -m pip install -r requirements.txt
python nn/train.py --data data/processed/mnist_1_9.npz --epochs 8 --seed 42
python data/generate_puzzles.py --out data/processed/puzzles --seed 42 --count 100 --givens 35 --misreads 2
python run_demo.py --mosaic data/processed/puzzles/mosaic_0000.png --weights nn/artifacts/digit_cnn.pt
python eval/run_experiments.py --seeds 5 --puzzles-per-level 100 --max-expansions 1000 --out results/final
pytest -q
```
