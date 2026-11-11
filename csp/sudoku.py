"""Sudoku CSP with Backtracking, Forward Checking, and AC-3.

The three inference levels are independently switchable:
    BT: use_fc=False, use_ac3=False
    BT+FC: use_fc=True, use_ac3=False
    BT+FC+AC3: use_fc=True, use_ac3=True
"""
from collections import deque
import numpy as np

CELLS = [(r, c) for r in range(9) for c in range(9)]
NEIGHBORS = {}
for r, c in CELLS:
    peers = {(r, j) for j in range(9) if j != c}
    peers |= {(i, c) for i in range(9) if i != r}
    br, bc = (r // 3) * 3, (c // 3) * 3
    peers |= {(i, j) for i in range(br, br + 3) for j in range(bc, bc + 3) if (i, j) != (r, c)}
    NEIGHBORS[(r, c)] = peers


class SudokuCSP:
    def __init__(self, grid, use_fc=True, use_ac3=True):
        arr = np.asarray(grid)
        if arr.shape != (9, 9):
            raise ValueError("grid must be 9x9")
        if not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr)):
            raise ValueError("grid must contain finite numeric values")
        if not np.all(arr == np.floor(arr)):
            raise ValueError("grid values must be integers")
        self.grid = arr.astype(int, copy=True)
        self.use_fc, self.use_ac3 = bool(use_fc), bool(use_ac3)
        self.stats = {"assignments": 0, "backtracks": 0, "pruned": 0, "nodes": 0}
        self.domains = {
            cell: ({int(self.grid[cell])} if self.grid[cell] else set(range(1, 10)))
            for cell in CELLS
        }
        self.consistent = self._initial_consistency()
        if self.consistent and self.use_ac3:
            self.consistent = self.ac3(self.domains)

    def _initial_consistency(self):
        for cell in CELLS:
            value = int(self.grid[cell])
            if value < 0 or value > 9:
                return False
        # With AC-3 enabled, let arc revision detect duplicate singleton givens
        # by emptying an unsupported domain. BT/BT+FC perform this check here.
        if not self.use_ac3:
            for cell in CELLS:
                value = int(self.grid[cell])
                if value and any(int(self.grid[p]) == value for p in NEIGHBORS[cell]):
                    return False
        return True

    def revise(self, domains, x, y):
        """Enforce x != y: remove x=v only when y has no supporting value."""
        unsupported = {v for v in domains[x] if not any(v != w for w in domains[y])}
        if unsupported:
            domains[x].difference_update(unsupported)
            self.stats["pruned"] += len(unsupported)
            return True
        return False

    def ac3(self, domains):
        queue = deque((x, y) for x in CELLS for y in NEIGHBORS[x])
        while queue:
            x, y = queue.popleft()
            if self.revise(domains, x, y):
                if not domains[x]:
                    return False
                for z in NEIGHBORS[x] - {y}:
                    queue.append((z, x))
        return all(domains[cell] for cell in CELLS)

    @staticmethod
    def _select(domains):
        """Dynamic MRV, including values ruled out by already assigned peers.

        Dynamic candidate calculation is essential for the pure-BT baseline:
        it gives BT a sensible variable ordering without doing forward pruning.
        """
        choices = []
        for cell in CELLS:
            if len(domains[cell]) == 1:
                continue
            used = {next(iter(domains[p])) for p in NEIGHBORS[cell] if len(domains[p]) == 1}
            candidates = domains[cell] - used
            degree = sum(len(domains[n]) > 1 for n in NEIGHBORS[cell])
            choices.append((len(candidates), -degree, cell, candidates))
        if not choices:
            return None, None
        _, _, cell, candidates = min(choices, key=lambda item: (item[0], item[1], item[2]))
        return cell, candidates

    def solve(self, require_unique=False):
        """Return a solution, or (solution, count) in uniqueness mode.

        Uniqueness mode stops after finding two solutions. A puzzle is accepted
        only when exactly one solution was found.
        """
        if not self.consistent:
            return (None, 0) if require_unique else None
        solutions = []
        limit = 2 if require_unique else 1

        def backtrack(domains):
            if len(solutions) >= limit:
                return
            self.stats["nodes"] += 1
            cell, candidates = self._select(domains)
            if cell is None:
                # Defensive complete-assignment validation, including BT-only mode.
                assignment = {c: next(iter(domains[c])) for c in CELLS}
                if all(assignment[x] != assignment[y] for x in CELLS for y in NEIGHBORS[x]):
                    solutions.append(assignment)
                return
            if not candidates:
                self.stats["backtracks"] += 1
                return
            for value in sorted(candidates):
                self.stats["assignments"] += 1
                # Even pure BT must reject a value conflicting with an assigned peer.
                if any(len(domains[p]) == 1 and value in domains[p] for p in NEIGHBORS[cell]):
                    self.stats["backtracks"] += 1
                    continue
                child = {c: set(vs) for c, vs in domains.items()}
                child[cell] = {value}
                ok = True
                if self.use_fc:
                    for peer in NEIGHBORS[cell]:
                        if value in child[peer]:
                            child[peer].remove(value)
                            self.stats["pruned"] += 1
                            if not child[peer]:
                                ok = False
                                break
                if ok and self.use_ac3:
                    ok = self.ac3(child)
                if ok:
                    backtrack(child)
                else:
                    self.stats["backtracks"] += 1
                if len(solutions) >= limit:
                    return
            if len(solutions) == 0:
                self.stats["backtracks"] += 1

        backtrack({c: set(v) for c, v in self.domains.items()})
        if not solutions:
            return (None, 0) if require_unique else None
        solution = np.array([[solutions[0][(r, c)] for c in range(9)] for r in range(9)], dtype=int)
        return (solution, len(solutions)) if require_unique else solution


def solve_sudoku(grid, use_fc=True, use_ac3=True, require_unique=False):
    return SudokuCSP(grid, use_fc, use_ac3).solve(require_unique=require_unique)
