# P7 Presentation Plan

## 10-minute presentation

| Time | Slide | Content |
|---:|---|---|
| 0:00–1:00 | 1. Problem | Handwritten Sudoku, misread digits, why duplicate clues expose errors |
| 1:00–2:00 | 2. Data | MNIST 28×28, digits 1–9, train/validation/test split, generated puzzle/mosaic distinction |
| 2:00–3:00 | 3. NN | CNN architecture, probability output, temperature calibration, Logistic Regression baseline |
| 3:00–4:00 | 4. Search | State, actions, nonnegative cost, UCS vs Greedy vs A*, 3-correction budget |
| 4:00–5:00 | 5. CSP | Variables/domains/constraints, Backtracking, Forward Checking, AC-3 |
| 5:00–6:00 | 6. Integration | NN → AC-3 → Search → CSP; correction log; unreadable path |
| 6:00–7:15 | 7. Experiment 1 | 0–4 misreads, 100 puzzles/level, recovery and expanded/generated nodes |
| 7:15–8:15 | 8. Experiment 2 | BT vs BT+FC vs BT+FC+AC-3, 25/30/35 givens |
| 8:15–9:15 | 9. Experiment 3 | NN-guided vs no correction vs random; exact recovery |
| 9:15–10:00 | 10. Limits/conclusion | Synthetic mosaics, no photo-grid localization, next steps |

Use only measured classifier metrics from `nn/artifacts/all_metrics.json` and experiment values from `results/summary.json`. Label synthetic controlled-probability results clearly.

## 5-minute live demo

1. **0:00–1:00 — Show the input.** Run `python run_demo.py --method ucs --max-corrections 3`. Explain that this demo uses a documented synthetic probability adapter because it is reproducible before CNN training.
2. **1:00–2:00 — Show Search trace.** Point to `expanded`, `generated`, and the correction log. Explain why the two wrong readings are inexpensive to replace.
3. **2:00–3:00 — Show CSP pruning.** Run `python eval/run_experiments.py --seeds 5 --puzzles-per-level 100 --max-expansions 1000 --out results` beforehand; show `experiment_2_nodes.png` and compare BT, BT+FC, and BT+FC+AC-3.
4. **3:00–4:00 — Show NN on/off.** If trained weights are available, run `python run_demo.py --mosaic data/processed/puzzles/mosaic_0000.png --weights nn/artifacts/digit_cnn.pt`. Contrast this with the controlled synthetic mode; never label synthetic probabilities as CNN output.
5. **4:00–5:00 — Failure handling.** Set a low correction budget or use an impossible duplicate-given test and show the `unreadable` response plus least-confident cells.

## Demo preparation checklist

- Run `pytest -q` and record the result.
- Run the experiments and verify all CSV/JSON/PNG outputs exist.
- Train the CNN and Logistic Regression baseline if network access and compute are available.
- Keep the generated mosaic, model weights, metrics and exact commands ready.
- Do not claim real-photo recognition; the current grid is a synthetic known-layout mosaic.
