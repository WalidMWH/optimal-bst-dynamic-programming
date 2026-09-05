"""Timing experiments for the optimal binary search tree dynamic program.

Random instances of growing size are timed, the DP fill separately from the
reconstruction, and the summarised results go to a CSV file and a log-log plot
drawn against a reference line proportional to n cubed. A least-squares fit of
log time against log n gives the empirical exponent, expected near 3.
"""

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

# The modules in src/ import each other by plain module name so that each one
# stays runnable on its own, so src/ has to be on the path before they load.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from baseline import tree_height  # noqa: E402
from obst import build_dp_tables  # noqa: E402
from tree import build_tree  # noqa: E402
from validation import generate_random  # noqa: E402

DEFAULT_SIZES = [5, 10, 20, 50, 100, 200, 400]
DEFAULT_REPEATS = 5
DEFAULT_SEED = 42


def inner_loop_count(n: int) -> int:
    """Exact number of candidate roots the DP evaluates over all subproblems."""
    return n * (n + 1) * (n + 2) // 6


def time_single_run(n: int, seed: int) -> dict:
    """Time one random instance, measuring the DP fill and the reconstruction apart."""
    keys, probabilities = generate_random(n, seed)

    dp_start = time.perf_counter()
    cost, root_table = build_dp_tables(probabilities)
    dp_seconds = time.perf_counter() - dp_start

    build_start = time.perf_counter()
    tree = build_tree(keys, probabilities, root_table, 1, n)
    build_seconds = time.perf_counter() - build_start

    return {
        "n": n,
        "seed": seed,
        "dp_seconds": dp_seconds,
        "build_seconds": build_seconds,
        "total_seconds": dp_seconds + build_seconds,
        "expected_cost": cost[1][n],
        "height": tree_height(tree),
    }


def run_experiment(sizes: list[int], repeats: int, seed: int) -> list[dict]:
    """Time every size over several random instances and summarise each size."""
    results: list[dict] = []

    for n in sizes:
        # Consecutive seeds: every trial is a different instance, yet the whole
        # experiment repeats exactly.
        trials = [time_single_run(n, seed + trial) for trial in range(repeats)]
        operations = inner_loop_count(n)
        mean_dp = statistics.fmean(trial["dp_seconds"] for trial in trials)

        results.append(
            {
                "n": n,
                "repeats": repeats,
                "mean_dp_seconds": mean_dp,
                # The minimum is the more robust estimator of the two: scheduler
                # noise can only ever add time to a run, never take any away.
                "min_dp_seconds": min(trial["dp_seconds"] for trial in trials),
                "mean_build_seconds": statistics.fmean(
                    trial["build_seconds"] for trial in trials
                ),
                "mean_total_seconds": statistics.fmean(
                    trial["total_seconds"] for trial in trials
                ),
                "mean_expected_cost": statistics.fmean(
                    trial["expected_cost"] for trial in trials
                ),
                "mean_height": statistics.fmean(trial["height"] for trial in trials),
                "inner_loop_count": operations,
                "time_per_operation": mean_dp / operations,
            }
        )

    return results


def write_csv(results: list[dict], path: str) -> None:
    """Write one row per input size, creating the parent directory if needed."""
    if not results:
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def estimate_exponent(results: list[dict]) -> float:
    """Least-squares slope of log time against log n, the empirical growth exponent."""
    # A timing of exactly zero has no logarithm; those sizes ran faster than the
    # clock can resolve and carry no information about the slope.
    points = [
        (math.log(row["n"]), math.log(row["mean_dp_seconds"]))
        for row in results
        if row["mean_dp_seconds"] > 0.0
    ]

    if len(points) < 2:
        return float("nan")

    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in points)
    variance = sum((x - mean_x) ** 2 for x, _ in points)

    if variance == 0.0:
        return float("nan")

    return covariance / variance


def plot_results(results: list[dict], path: str) -> bool:
    """Draw the log-log timing plot against an n cubed reference line and save it."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # saving to a file needs no display
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot skipped: matplotlib is not installed (pip install matplotlib)")
        return False

    measured = [row for row in results if row["mean_dp_seconds"] > 0.0]
    if not measured:
        print("plot skipped: every measured time was below the clock resolution")
        return False

    sizes = [row["n"] for row in measured]
    times = [row["mean_dp_seconds"] for row in measured]

    # The reference is anchored at the largest measured point so the two curves
    # meet there and only their slopes are left to compare.
    anchor = max(measured, key=lambda row: row["n"])
    reference = [
        anchor["mean_dp_seconds"] * (n / anchor["n"]) ** 3 for n in sizes
    ]

    figure, axes = plt.subplots(figsize=(7, 5))
    axes.plot(sizes, times, marker="o", label="measured DP table fill")
    axes.plot(sizes, reference, linestyle="--", label="reference proportional to n^3")
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("number of keys n")
    axes.set_ylabel("mean DP table fill time (s)")
    axes.set_title("Optimal BST dynamic programming: running time against n")
    axes.grid(True, which="both", linewidth=0.3)
    axes.legend()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return True


def build_argument_parser() -> argparse.ArgumentParser:
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Measure how the optimal BST dynamic program scales with n."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        metavar="N",
        help="input sizes to time",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        metavar="R",
        help=f"trials per size (default {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="S",
        help=f"base seed for the random instances (default {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--csv",
        default="results/timings.csv",
        metavar="PATH",
        help="where to write the results table",
    )
    parser.add_argument(
        "--plot",
        default="results/complexity.png",
        metavar="PATH",
        help="where to write the log-log plot",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip the plot entirely"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the experiment, report the fitted exponent and write the CSV and plot."""
    args = build_argument_parser().parse_args(argv)

    print(
        f"timing {len(args.sizes)} sizes, {args.repeats} trials each, "
        f"base seed {args.seed}"
    )

    # n = 400 alone takes a few seconds, so the default run is on the order of
    # half a minute. Sizes are timed one at a time so progress can be shown
    # instead of the program sitting silent until everything is done.
    results: list[dict] = []
    for n in args.sizes:
        results.extend(run_experiment([n], args.repeats, args.seed))
        row = results[-1]
        print(
            f"  n = {row['n']:>5}   mean DP {row['mean_dp_seconds']:.6f} s"
            f"   min DP {row['min_dp_seconds']:.6f} s"
        )
    print()

    header = (
        f"{'n':>6}{'mean DP (s)':>14}{'min DP (s)':>14}"
        f"{'mean build (s)':>16}{'inner loops':>14}{'us per loop':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['n']:>6}{row['mean_dp_seconds']:>14.6f}"
            f"{row['min_dp_seconds']:>14.6f}{row['mean_build_seconds']:>16.6f}"
            f"{row['inner_loop_count']:>14}"
            f"{row['time_per_operation'] * 1e6:>13.4f}"
        )
    print()

    exponent = estimate_exponent(results)
    print(f"{'fitted exponent':<22}= {exponent:.3f}")
    print(f"{'theoretical exponent':<22}= 3")
    if math.isnan(exponent):
        print("too few measurable timings to judge the growth rate")
    elif abs(exponent - 3.0) <= 0.3:
        print(
            "the fitted exponent is within 0.3 of 3, so the measurements are "
            "consistent with cubic growth"
        )
    else:
        print(
            "the fitted exponent is more than 0.3 from 3, so the measurements are "
            "not cleanly cubic; the small sizes are dominated by interpreter "
            "overhead and flatten the fit"
        )
    print()

    write_csv(results, args.csv)
    print(f"wrote {args.csv}")

    if not args.no_plot and plot_results(results, args.plot):
        print(f"wrote {args.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())