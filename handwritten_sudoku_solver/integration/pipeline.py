"""End-to-end NN -> AC-3 -> correction Search -> CSP pipeline."""
import numpy as np
from csp.sudoku import SudokuCSP
from search.correction import correction_search


def controlled_probabilities(read_grid, solution, injected, seed=0):
    """Deterministic synthetic NN-score adapter used only for reproducible demos.

    It deliberately models an ambiguous read (52% wrong / 45% true) for each
    injected error. Replace this adapter with nn.inference.predict_mosaic when
    trained model weights are available.
    """
    rng = np.random.default_rng(seed)
    probs = np.full((9, 9, 9), 0.001, dtype=np.float64)
    injected_map = {(x["row"], x["col"]): x for x in injected}
    for r in range(9):
        for c in range(9):
            digit = int(read_grid[r, c])
            if digit == 0:
                probs[r, c, :] = 1.0 / 9
            elif (r, c) in injected_map:
                true = int(injected_map[(r, c)]["true_digit"])
                probs[r, c, digit - 1] = 0.52
                probs[r, c, true - 1] = 0.45
            else:
                probs[r, c, digit - 1] = 0.97
            probs[r, c, :] /= probs[r, c, :].sum()
    return probs


def run_pipeline(read_grid, probabilities, max_corrections=3, method="ucs", require_unique=False):
    """Correct a read grid, then return a CSP solution and a transparent edit log."""
    read_grid = np.asarray(read_grid, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if read_grid.shape != (9, 9) or probabilities.shape != (9, 9, 9):
        raise ValueError("expected a 9x9 grid and 9x9x9 class probabilities")

    # AC-3 performs early conflict detection before any correction search.
    initial = SudokuCSP(read_grid, use_fc=True, use_ac3=True)
    if initial.consistent:
        first_solver = SudokuCSP(read_grid, use_fc=True, use_ac3=True)
        first = first_solver.solve(require_unique=require_unique)
        solved = first[0] if require_unique else first
        unique_count = first[1] if require_unique else None
        if solved is not None and (not require_unique or unique_count == 1):
            return {"status": "solved_without_correction", "grid": solved,
                    "corrections": [], "search": {"cost": 0.0, "expanded": 0, "generated": 1},
                    "csp_stats": first_solver.stats}

    def csp_check(candidate):
        solver = SudokuCSP(candidate, use_fc=True, use_ac3=True)
        return solver

    result = correction_search(read_grid, probabilities, method=method,
                               max_corrections=max_corrections, top_k=3,
                               csp_factory=csp_check)
    if result["grid"] is None:
        return {"status": "unreadable", "grid": None, "corrections": [],
                "search": {k: result[k] for k in ("cost", "expanded", "generated")},
                "least_confident_cells": _least_confident(read_grid, probabilities)}

    solver = SudokuCSP(result["grid"], use_fc=True, use_ac3=True)
    final = solver.solve(require_unique=require_unique)
    solved = final[0] if require_unique else final
    unique_count = final[1] if require_unique else None
    if solved is None or (require_unique and unique_count != 1):
        return {"status": "unreadable", "grid": None, "corrections": result["corrections"],
                "search": {k: result[k] for k in ("cost", "expanded", "generated")},
                "least_confident_cells": _least_confident(read_grid, probabilities)}
    return {"status": "solved", "grid": solved, "corrections": result["corrections"],
            "search": {k: result[k] for k in ("cost", "expanded", "generated")},
            "csp_stats": solver.stats}


def _least_confident(grid, probabilities, n=5):
    cells = []
    for r in range(9):
        for c in range(9):
            digit = int(grid[r, c])
            if digit:
                cells.append((float(probabilities[r, c, digit - 1]), r, c, digit))
    return [{"row": r, "col": c, "read_digit": d, "confidence": conf}
            for conf, r, c, d in sorted(cells)[:n]]
