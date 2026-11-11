"""Run a reproducible synthetic demo or infer from a generated mosaic with a trained CNN."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data.sudoku_generator import make_puzzle
from integration.pipeline import controlled_probabilities, run_pipeline


def make_conflicting_demo(seed=7):
    solution, puzzle = make_puzzle(seed, givens=35, require_unique=True)
    read = puzzle.copy(); injected = []
    for r in range(9):
        for c in range(9):
            if read[r, c] != 0:
                continue
            peers = set(int(x) for x in read[r, :] if x)
            peers |= set(int(x) for x in read[:, c] if x)
            br, bc = (r // 3) * 3, (c // 3) * 3
            peers |= set(int(x) for x in read[br:br+3, bc:bc+3].ravel() if x)
            options = sorted(x for x in peers if x != int(solution[r, c]))
            if options:
                wrong = options[0]
                read[r, c] = wrong
                injected.append({"row": r, "col": c, "true_digit": int(solution[r, c]), "read_digit": wrong})
            if len(injected) == 2:
                break
        if len(injected) == 2:
            break
    if len(injected) != 2:
        raise RuntimeError("could not create two conflicting demo errors")
    return solution, puzzle, read, injected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mosaic", default=None, help="252x252 generated mosaic to classify with a trained CNN")
    parser.add_argument("--weights", default="nn/artifacts/digit_cnn.pt")
    parser.add_argument("--calibration", default="nn/artifacts/calibration.json")
    parser.add_argument("--method", choices=("ucs", "greedy", "astar"), default="ucs")
    parser.add_argument("--max-corrections", type=int, default=3)
    parser.add_argument("--require-unique", action="store_true")
    args = parser.parse_args()

    if args.mosaic:
        from nn.inference import predict_mosaic
        read, probs = predict_mosaic(args.mosaic, args.weights, args.calibration)
        solution = None
        print("Input mode: REAL CNN inference on a generated MNIST mosaic")
    else:
        solution, _, read, injected = make_conflicting_demo()
        probs = controlled_probabilities(read, solution, injected, seed=7)
        print("Input mode: controlled synthetic probabilities (not trained CNN output)")
        print("Injected errors:", json.dumps(injected, indent=2))

    result = run_pipeline(read, probs, max_corrections=args.max_corrections,
                          method=args.method, require_unique=args.require_unique)
    print("INPUT GRID (0 = blank):\n", read)
    print("PIPELINE STATUS:", result["status"])
    print("CORRECTION LOG:", json.dumps(result.get("corrections", []), indent=2))
    print("SEARCH:", json.dumps(result.get("search", {}), indent=2))
    print("CSP COUNTERS:", json.dumps(result.get("csp_stats", {}), indent=2))
    if result.get("grid") is not None:
        print("SOLVED GRID:\n", result["grid"])
    else:
        print("LEAST CONFIDENT CELLS:", json.dumps(result.get("least_confident_cells", []), indent=2))

    Path("results").mkdir(exist_ok=True)
    panels = [read, result.get("grid")]
    titles = ["Read grid", "CSP solution"]
    if solution is not None:
        panels.append(solution); titles.append("Known generated solution (evaluation only)")
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for ax, grid, title in zip(axes, panels, titles):
        ax.imshow(grid if grid is not None else np.zeros((9, 9)), vmin=0, vmax=9, cmap="Blues")
        ax.set_title(title); ax.set_xticks(range(9)); ax.set_yticks(range(9)); ax.grid()
    fig.tight_layout(); fig.savefig("results/demo_pipeline.png", dpi=160); plt.close(fig)
    print("Saved results/demo_pipeline.png")


if __name__ == "__main__":
    main()
