"""UCS, Greedy Best-First and A* over sets of digit-reading corrections."""
import heapq
import itertools
import math
import numpy as np
from csp.sudoku import CELLS, NEIGHBORS, SudokuCSP


def conflicts(grid):
    """Cells participating in at least one duplicate given in row/column/box."""
    bad = set()
    for cell in CELLS:
        value = int(grid[cell])
        if value and any(int(grid[peer]) == value for peer in NEIGHBORS[cell]):
            bad.add(cell)
    return bad


def _change_cost(cell, old, new, probs):
    """Nonnegative increase in negative-log likelihood for changing old to new."""
    p = probs[cell[0], cell[1]]
    return max(0.0, math.log(max(float(p[old - 1]), 1e-12) / max(float(p[new - 1]), 1e-12)))


def _options(cell, old, probs, top_k=3):
    p = np.asarray(probs[cell[0], cell[1]], dtype=float)
    order = np.argsort(-p, kind="stable")
    return [int(i + 1) for i in order if i + 1 != old][:top_k]


def correction_heuristic(grid, probs, top_k=3):
    """Admissible lower bound from a greedy matching of duplicate-conflict edges.

    Every selected edge has disjoint endpoints, and at least one endpoint of
    each edge must change. Summing the cheapest available endpoint-change cost
    therefore cannot overestimate the cost of resolving all selected edges.
    """
    bad = conflicts(grid)
    edges = sorted((x, y) for x in bad for y in NEIGHBORS[x]
                   if y in bad and grid[x] == grid[y] and x < y)
    used, lower_bound = set(), 0.0
    for x, y in edges:
        if x in used or y in used:
            continue
        def cheapest(cell):
            old = int(grid[cell])
            return min((_change_cost(cell, old, new, probs)
                        for new in _options(cell, old, probs, top_k)), default=0.0)
        lower_bound += min(cheapest(x), cheapest(y))
        used.update((x, y))
    return lower_bound


def correction_search(read_grid, probs, method="ucs", max_corrections=3,
                      top_k=3, csp_factory=None, max_expansions=50000):
    """Find the least-cost solvable corrected grid within the edit budget.

    Search states are complete 9x9 readings; actions edit one currently
    conflicting cell to one of its top-k alternative NN classes. UCS is the
    optimal-cost reference. Greedy and A* are comparison strategies.
    """
    method = method.lower()
    if method not in {"ucs", "greedy", "astar"}:
        raise ValueError("method must be 'ucs', 'greedy', or 'astar'")
    if max_corrections < 0 or top_k < 1 or max_expansions < 1:
        raise ValueError("max_corrections must be >=0, top_k >=1 and max_expansions >=1")
    grid = np.asarray(read_grid, dtype=int)
    probabilities = np.asarray(probs, dtype=float)
    if grid.shape != (9, 9) or probabilities.shape != (9, 9, 9):
        raise ValueError("expected grid shape (9,9) and probabilities shape (9,9,9)")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if np.any(probabilities.sum(axis=2) <= 0):
        raise ValueError("each cell must have positive total probability")
    probabilities = probabilities / probabilities.sum(axis=2, keepdims=True)

    start = tuple(grid.reshape(-1).tolist())
    counter = itertools.count()

    def grid_of(state):
        return np.asarray(state, dtype=int).reshape(9, 9)

    def solvable(candidate):
        if conflicts(candidate):
            return False
        solver = csp_factory(candidate) if csp_factory else SudokuCSP(candidate, True, True)
        return solver.solve() is not None

    if solvable(grid):
        return {"grid": grid.copy(), "corrections": [], "cost": 0.0,
                "expanded": 0, "generated": 1}

    start_h = correction_heuristic(grid, probabilities, top_k)
    heap = [(start_h if method == "astar" else 0.0, next(counter), 0.0, start, [])]
    best_cost = {start: 0.0}
    expanded, generated = 0, 1

    while heap and expanded < max_expansions:
        _, _, cost_so_far, state, log = heapq.heappop(heap)
        if cost_so_far > best_cost.get(state, float("inf")) + 1e-12:
            continue
        expanded += 1
        current = grid_of(state)
        if solvable(current):
            return {"grid": current, "corrections": log, "cost": cost_so_far,
                    "expanded": expanded, "generated": generated}
        if len(log) >= max_corrections:
            continue
        already_corrected = {(item["row"], item["col"]) for item in log}
        for cell in sorted(conflicts(current)):
            # A correction state is a set of edits relative to the first read:
            # each cell may be changed at most once.
            if cell in already_corrected:
                continue
            old = int(current[cell])
            for new in _options(cell, old, probabilities, top_k):
                child = current.copy()
                child[cell] = new
                child_state = tuple(child.reshape(-1).tolist())
                step_cost = _change_cost(cell, old, new, probabilities)
                new_cost = cost_so_far + step_cost
                if new_cost + 1e-12 >= best_cost.get(child_state, float("inf")):
                    continue
                best_cost[child_state] = new_cost
                generated += 1
                edit = {"row": cell[0], "col": cell[1], "old": old,
                        "new": new, "cost": step_cost}
                h = correction_heuristic(child, probabilities, top_k)
                priority = h if method == "greedy" else (new_cost + h if method == "astar" else new_cost)
                heapq.heappush(heap, (priority, next(counter), new_cost,
                                    child_state, log + [edit]))

    return {"grid": None, "corrections": [], "cost": float("inf"),
            "expanded": expanded, "generated": generated,
            "budget_exhausted": bool(heap and expanded >= max_expansions)}
