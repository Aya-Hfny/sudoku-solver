"""Reproducible evaluation of all three required P7 experiments."""
import argparse
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data.sudoku_generator import make_puzzle
from csp.sudoku import SudokuCSP, NEIGHBORS
from search.correction import correction_search, conflicts
from integration.pipeline import controlled_probabilities


def inject_conflicting_errors(solution, puzzle, n, seed):
    """Inject n wrong digits into blank cells, each guaranteed to duplicate a peer."""
    rng = np.random.default_rng(seed)
    read = puzzle.copy()
    eligible = []
    for r in range(9):
        for c in range(9):
            if read[r, c] != 0:
                continue
            peers = set(int(x) for x in read[r, :] if x)
            peers |= set(int(x) for x in read[:, c] if x)
            br, bc = (r // 3) * 3, (c // 3) * 3
            peers |= set(int(x) for x in read[br:br+3, bc:bc+3].ravel() if x)
            choices = sorted(v for v in peers if v != int(solution[r, c]))
            if choices:
                eligible.append((r, c, choices))
    rng.shuffle(eligible)
    log = []
    for r, c, choices in eligible[:n]:
        true = int(solution[r, c])
        wrong = int(rng.choice(choices))
        read[r, c] = wrong
        log.append({"row": r, "col": c, "true_digit": true, "read_digit": wrong})
    if len(log) != n:
        raise RuntimeError(f"could inject only {len(log)} of {n} requested conflicting errors")
    return read, log


def exact_recovery(candidate, solution):
    """A corrected clue grid recovers the exact solution iff every given agrees.

    Search has already required CSP solvability. Since the generated source
    puzzle is unique, agreement of every remaining given with its known answer
    proves that the completed solution is the generated ground truth.
    """
    return candidate is not None and bool(np.all((candidate == 0) | (candidate == solution)))


def random_correction(read, probabilities, seed, max_corrections=3):
    """Random baseline: random conflicting cell and random alternative digit."""
    rng = np.random.default_rng(seed)
    current = read.copy()
    log = []
    for _ in range(max_corrections + 1):
        if not conflicts(current):
            return current, log
        bad = sorted(conflicts(current))
        if not bad or len(log) >= max_corrections:
            break
        cell = bad[int(rng.integers(len(bad)))]
        old = int(current[cell])
        alternatives = [v for v in range(1, 10) if v != old]
        new = int(rng.choice(alternatives))
        current[cell] = new
        log.append({"row": cell[0], "col": cell[1], "old": old, "new": new})
    return None, log


def write_summary(rows, out):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["experiment"], row["setting"], row["method"])
        grouped[key].append(row)
    summary = []
    for (experiment, setting, method), values in sorted(grouped.items()):
        item = {"experiment": experiment, "setting": setting, "method": method,
                "n": len(values)}
        for metric in ("recovery", "expanded", "generated"):
            vals = [float(x[metric]) for x in values]
            item[f"{metric}_mean"] = statistics.mean(vals)
            item[f"{metric}_sd"] = statistics.pstdev(vals)
        summary.append(item)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_experiment_1(rows, out):
    methods = ("ucs", "greedy", "astar")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        xs, ys = [], []
        for errors in range(5):
            vals = [x["recovery"] for x in rows if x["experiment"] == 1 and x["setting"] == errors and x["method"] == method]
            xs.append(errors); ys.append(np.mean(vals))
        ax.plot(xs, ys, marker="o", label=method.upper())
    ax.set(xlabel="Injected conflicting misreads", ylabel="Exact solution recovery rate", ylim=(-0.03, 1.03), title="Experiment 1: Search strategy")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "experiment_1_recovery.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        xs, ys = [], []
        for errors in range(5):
            vals = [x["expanded"] for x in rows if x["experiment"] == 1 and x["setting"] == errors and x["method"] == method]
            xs.append(errors); ys.append(np.mean(vals))
        ax.plot(xs, ys, marker="o", label=method.upper())
    ax.set(xlabel="Injected conflicting misreads", ylabel="Mean expanded nodes", title="Experiment 1: Search effort")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "experiment_1_nodes.png", dpi=160); plt.close(fig)


def plot_experiment_2(rows, out):
    methods = ("BT", "BT+FC", "BT+FC+AC3")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        xs, ys = [], []
        for givens in (25, 30, 35):
            vals = [x["expanded"] for x in rows if x["experiment"] == 2 and x["setting"] == givens and x["method"] == method]
            xs.append(givens); ys.append(np.mean(vals))
        ax.plot(xs, ys, marker="o", label=method)
    ax.set(xlabel="Number of givens", ylabel="Mean search nodes", title="Experiment 2: CSP inference")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "experiment_2_nodes.png", dpi=160); plt.close(fig)


def plot_experiment_3(rows, out):
    methods = ("NN-guided", "No correction", "Random correction")
    keys = ("nn_guided", "no_correction", "random")
    vals = [np.mean([x["recovery"] for x in rows if x["experiment"] == 3 and x["method"] == key]) for key in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, vals); ax.set_ylim(0, 1.05)
    ax.set(ylabel="Exact solution recovery rate", title="Experiment 3: Correction policy")
    fig.tight_layout(); fig.savefig(out / "experiment_3_recovery.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="results")
    parser.add_argument("--max-corrections", type=int, default=4)
    parser.add_argument("--puzzles-per-level", type=int, default=100)
    parser.add_argument("--max-expansions", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0,
                        help="starting puzzle seed index; supports reproducible chunked 100-puzzle runs")
    parser.add_argument("--skip-csp-experiment", action="store_true",
                        help="skip experiment 2 when running later chunks")
    parser.add_argument("--chunked", action="store_true",
                        help="allow a partial seed range intended to be merged into a 100-puzzle final run")
    parser.add_argument("--pilot", action="store_true", help="allow fewer than 100 puzzles for a clearly labelled pilot run")
    args = parser.parse_args()
    if args.seeds < 5:
        parser.error("P7 experiments require at least 5 seeds; use --seeds 5 or more")
    if args.puzzles_per_level < 100 and not (args.pilot or args.chunked):
        parser.error("P7 exact-recovery evaluation requires at least 100 puzzles per misread level; use --pilot only for a pilot")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    correction_logs = []

    # Experiment 1: UCS vs Greedy vs A*, at 0,1,2,3,4 conflicting misreads.
    for seed in range(args.start_seed, args.start_seed + args.puzzles_per_level):
        if seed % 10 == 0:
            print(f"[Experiment 1] puzzle {seed + 1}/{args.puzzles_per_level}", flush=True)
        solution, puzzle = make_puzzle(100 + seed, givens=35, require_unique=True)
        for errors in range(5):
            read, injected = inject_conflicting_errors(solution, puzzle, errors, seed + errors * 1000)
            probs = controlled_probabilities(read, solution, injected, seed)
            for method in ("ucs", "greedy", "astar"):
                result = correction_search(read, probs, method=method,
                    max_corrections=args.max_corrections, top_k=3,
                    csp_factory=lambda g: SudokuCSP(g, True, True), max_expansions=args.max_expansions)
                recovered = int(exact_recovery(result["grid"], solution))
                rows.append({"experiment": 1, "seed": seed, "setting": errors,
                    "method": method, "expanded": result["expanded"],
                    "generated": result["generated"], "recovery": recovered})
                correction_logs.append({"experiment": 1, "seed": seed, "misread_level": errors,
                    "method": method, "injected_misreads": injected,
                    "corrections": result["corrections"], "exact_recovery": recovered,
                    "budget_exhausted": result.get("budget_exhausted", False)})
        print(f"[Experiment 1] completed puzzle {seed + 1}", flush=True)

    # Experiment 2: compare inference levels on unique 25/30/35-given puzzles.
    for seed in ([] if args.skip_csp_experiment else range(args.seeds)):
        print(f"[Experiment 2] seed {seed + 1}/{args.seeds}", flush=True)
        for givens in (25, 30, 35):
            solution, puzzle = make_puzzle(500 + seed * 10 + givens, givens, True)
            for fc, ac3, label in ((False, False, "BT"), (True, False, "BT+FC"), (True, True, "BT+FC+AC3")):
                solver = SudokuCSP(puzzle, use_fc=fc, use_ac3=ac3)
                solved = solver.solve()
                rows.append({"experiment": 2, "seed": seed, "setting": givens,
                    "method": label, "expanded": solver.stats["nodes"],
                    "generated": solver.stats["assignments"],
                    "recovery": int(solved is not None and np.array_equal(solved, solution))})

    # Experiment 3: NN-guided correction vs no correction vs random policy.
    for seed in range(args.start_seed, args.start_seed + args.puzzles_per_level):
        if seed % 10 == 0:
            print(f"[Experiment 3] puzzle {seed + 1}/{args.puzzles_per_level}", flush=True)
        solution, puzzle = make_puzzle(900 + seed, givens=35, require_unique=True)
        read, injected = inject_conflicting_errors(solution, puzzle, 2, seed + 33)
        probs = controlled_probabilities(read, solution, injected, seed)
        guided = correction_search(read, probs, method="ucs", max_corrections=3,
                                   csp_factory=lambda g: SudokuCSP(g, True, True), max_expansions=args.max_expansions)
        guided_recovery = int(exact_recovery(guided["grid"], solution))
        rows.append({"experiment": 3, "seed": seed, "setting": 2, "method": "nn_guided",
            "expanded": guided["expanded"], "generated": guided["generated"],
            "recovery": guided_recovery})
        correction_logs.append({"experiment": 3, "seed": seed, "misread_level": 2,
            "method": "nn_guided", "injected_misreads": injected,
            "corrections": guided["corrections"], "exact_recovery": guided_recovery,
            "budget_exhausted": guided.get("budget_exhausted", False)})
        rows.append({"experiment": 3, "seed": seed, "setting": 2, "method": "no_correction",
            "expanded": 0, "generated": 0,
            "recovery": int(exact_recovery(read, solution))})
        random_grid, random_log = random_correction(read, probs, seed, max_corrections=3)
        random_recovery = int(exact_recovery(random_grid, solution))
        rows.append({"experiment": 3, "seed": seed, "setting": 2, "method": "random",
            "expanded": 0, "generated": 0, "recovery": random_recovery})
        correction_logs.append({"experiment": 3, "seed": seed, "misread_level": 2,
            "method": "random", "injected_misreads": injected,
            "corrections": random_log, "exact_recovery": random_recovery,
            "budget_exhausted": False})

    fields = ["experiment", "seed", "setting", "method", "expanded", "generated", "recovery"]
    with (out / "experiment_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (out / "correction_logs.jsonl").open("w", encoding="utf-8") as handle:
        for item in correction_logs:
            handle.write(json.dumps(item) + "\n")
    summary = write_summary(rows, out)
    plot_experiment_1(rows, out); plot_experiment_2(rows, out); plot_experiment_3(rows, out)
    print(json.dumps({"runs": len(rows), "experiment_1_and_3_puzzles_per_level": args.puzzles_per_level,
                      "start_seed": args.start_seed, "csp_experiment_skipped": args.skip_csp_experiment, "chunked": args.chunked,
                      "experiment_2_seeds": args.seeds, "max_expansions_per_search": args.max_expansions,
                      "pilot": args.pilot, "summary_rows": len(summary), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
