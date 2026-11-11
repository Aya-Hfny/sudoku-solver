import numpy as np
import pytest
from search.correction import correction_search
from integration.pipeline import controlled_probabilities
from data.sudoku_generator import make_puzzle
from csp.sudoku import SudokuCSP


def one_error_case():
    solution, puzzle = make_puzzle(33, 35, True)
    for r in range(9):
        for c in range(9):
            if puzzle[r, c] == 0:
                peers = set(int(x) for x in puzzle[r, :] if x)
                peers |= set(int(x) for x in puzzle[:, c] if x)
                br, bc = (r // 3) * 3, (c // 3) * 3
                peers |= set(int(x) for x in puzzle[br:br+3, bc:bc+3].ravel() if x)
                wrongs = sorted(x for x in peers if x != int(solution[r, c]))
                if wrongs:
                    wrong = wrongs[0]; read = puzzle.copy(); read[r, c] = wrong
                    injected = [{"row": r, "col": c, "true_digit": int(solution[r, c]), "read_digit": wrong}]
                    return solution, read, injected, (r, c)
    raise AssertionError("no suitable blank cell")


@pytest.mark.parametrize("method", ["ucs", "greedy", "astar"])
def test_search_known_one_error_all_methods(method):
    solution, read, injected, cell = one_error_case()
    probs = controlled_probabilities(read, solution, injected, seed=3)
    result = correction_search(read, probs, method=method, max_corrections=3,
                               csp_factory=lambda g: SudokuCSP(g, True, True))
    assert result["grid"] is not None
    assert result["grid"][cell] == solution[cell]
    solved = SudokuCSP(result["grid"]).solve()
    assert np.array_equal(solved, solution)
    assert result["generated"] >= 1 and result["expanded"] >= 1


def test_search_rejects_bad_method():
    with pytest.raises(ValueError):
        correction_search(np.zeros((9, 9)), np.ones((9, 9, 9)), method="beam")
