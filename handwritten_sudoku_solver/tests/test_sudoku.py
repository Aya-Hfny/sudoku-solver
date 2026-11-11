import numpy as np
from csp.sudoku import SudokuCSP
from data.sudoku_generator import make_puzzle, make_solution

# Five normal puzzle cases / inference configurations.
def test_normal_easy_unique():
    solution, puzzle = make_puzzle(1, 35, True)
    assert np.array_equal(SudokuCSP(puzzle).solve(), solution)

def test_normal_medium_unique():
    solution, puzzle = make_puzzle(2, 30, True)
    assert np.array_equal(SudokuCSP(puzzle).solve(), solution)

def test_normal_backtracking_only():
    _, puzzle = make_puzzle(3, 45, True)
    solved = SudokuCSP(puzzle, use_fc=False, use_ac3=False).solve()
    assert solved is not None and np.all(solved >= 1) and np.all(solved <= 9)

def test_normal_forward_checking():
    solution, puzzle = make_puzzle(4, 35, True)
    assert np.array_equal(SudokuCSP(puzzle, use_fc=True, use_ac3=False).solve(), solution)

def test_normal_forward_checking_and_ac3():
    solution, puzzle = make_puzzle(5, 30, True)
    assert np.array_equal(SudokuCSP(puzzle, use_fc=True, use_ac3=True).solve(), solution)

# Three edge cases.
def test_edge_complete_valid_grid():
    solution = make_solution(8)
    assert np.array_equal(SudokuCSP(solution).solve(), solution)

def test_edge_nearly_complete_grid():
    solution = make_solution(9); puzzle = solution.copy(); puzzle[0, 0] = 0
    assert np.array_equal(SudokuCSP(puzzle).solve(), solution)

def test_edge_empty_grid_returns_valid_solution():
    solved = SudokuCSP(np.zeros((9, 9), dtype=int)).solve()
    assert solved is not None
    for r in range(9):
        assert set(solved[r]) == set(range(1, 10))
        assert set(solved[:, r]) == set(range(1, 10))

# Two impossible inputs.
def test_impossible_duplicate_givens():
    grid = np.zeros((9, 9), dtype=int); grid[0, 0] = 5; grid[0, 1] = 5
    assert SudokuCSP(grid, use_fc=True, use_ac3=False).solve() is None
    ac3_solver = SudokuCSP(grid, use_fc=True, use_ac3=True)
    assert ac3_solver.solve() is None
    assert ac3_solver.stats["pruned"] > 0

def test_impossible_out_of_range():
    grid = np.zeros((9, 9), dtype=int); grid[0, 0] = 10
    assert SudokuCSP(grid).solve() is None


def test_unique_mode_reports_two_for_empty_grid():
    solution, count = SudokuCSP(np.zeros((9, 9), dtype=int)).solve(require_unique=True)
    assert solution is not None and count == 2
