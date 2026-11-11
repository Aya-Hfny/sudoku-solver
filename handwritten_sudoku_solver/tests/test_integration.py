import numpy as np
from data.sudoku_generator import make_puzzle
from integration.pipeline import controlled_probabilities, run_pipeline


def test_pipeline_corrects_two_conflicting_reads_and_logs_edits():
    solution, puzzle = make_puzzle(77, 35, True)
    read = puzzle.copy(); injected = []
    for r in range(9):
        for c in range(9):
            if read[r, c] != 0:
                continue
            peers = set(int(x) for x in read[r, :] if x)
            peers |= set(int(x) for x in read[:, c] if x)
            br, bc = (r // 3) * 3, (c // 3) * 3
            peers |= set(int(x) for x in read[br:br+3, bc:bc+3].ravel() if x)
            choices = sorted(x for x in peers if x != int(solution[r, c]))
            if choices:
                wrong = choices[0]
                read[r, c] = wrong
                injected.append({"row": r, "col": c, "true_digit": int(solution[r, c]), "read_digit": wrong})
            if len(injected) == 2:
                break
        if len(injected) == 2:
            break
    probs = controlled_probabilities(read, solution, injected, seed=77)
    result = run_pipeline(read, probs, max_corrections=3, method="ucs")
    assert result["status"] == "solved"
    assert np.array_equal(result["grid"], solution)
    assert len(result["corrections"]) == 2
    assert all(set(edit) >= {"row", "col", "old", "new", "cost"} for edit in result["corrections"])


def test_pipeline_returns_unreadable_and_least_confident_cells():
    read = np.zeros((9, 9), dtype=int)
    read[0, 0] = 5; read[0, 1] = 5
    probs = np.full((9, 9, 9), 1 / 9, dtype=float)
    result = run_pipeline(read, probs, max_corrections=0, method="ucs")
    assert result["status"] == "unreadable"
    assert result["grid"] is None
    assert result["least_confident_cells"]
