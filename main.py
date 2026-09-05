# Command-line entry point:
# Reads the input, executes the dynamic programming algorithm, reconstructs the optimal tree, and reports the required outputs

import argparse
import sys
import time
from pathlib import Path

# The modules in src/ import each other by plain name, so src/ must be on the path first
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from baseline import compare_trees  # noqa: E402
from obst import build_dp_tables    # noqa: E402
from tree import (                  # noqa: E402
    build_tree,
    collect_depths,
    expected_cost_from_tree,
    render_tree,
)
from validation import (            # noqa: E402
    InvalidInputError,
    generate_random,
    load_from_file,
    read_interactive,
    validate,
)

MAX_TABLE_KEYS = 15

# Configures the application to accept the required inputs: number of keys, sorted keys, and probabilities
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the optimal binary search tree for a set of sorted keys."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", metavar="PATH", help="read the input from a file")
    source.add_argument(
        "--interactive", action="store_true", help="prompt for the input on stdin"
    )
    source.add_argument(
        "--random", type=int, metavar="N", help="generate N random keys"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed used by --random (default 42)"
    )
    # Triggers the requirement to construct a conventional BST and compare it with the Optimal BST
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also compare against conventional binary search trees",
    )
    parser.add_argument(
        "--no-tables", action="store_true", help="suppress the cost and root tables"
    )
    return parser

# Loads the provided input to compute the result from the supplied keys and successful-search probabilities
def load_input(args: argparse.Namespace) -> tuple[list[int], list[float]]:
    if args.file is not None:
        return load_from_file(args.file)
    if args.interactive:
        return read_interactive()
    if args.random is not None:
        return generate_random(args.random, args.seed)

    # Uses the example input provided in the project file
    keys = [0, 10, 20, 30, 40, 50]
    probabilities = [0.0, 0.10, 0.20, 0.40, 0.20, 0.10]
    validate(keys, probabilities)
    return keys, probabilities

# Displays the Dynamic Programming cost table and the root table as required for the project report
def render_upper_triangle(
    table: list[list[float]] | list[list[int]],
    n: int,
    width: int,
    is_float: bool,
) -> list[str]:
    label_width = 6
    lines = [
        " " * label_width + "".join(f"{j:>{width}}" for j in range(1, n + 1)),
        " " * label_width + "-" * (width * n),
    ]
    for i in range(1, n + 1):
        cells = []
        for j in range(1, n + 1):
            if j < i:
                cells.append(" " * width)
            elif is_float:
                cells.append(f"{table[i][j]:>{width}.4f}")
            else:
                cells.append(f"{table[i][j]:>{width}d}")
        lines.append(f"{i:>4} |" + "".join(cells))
    return lines

# Reports the input keys alongside their corresponding probabilities
def print_input_table(keys: list[int], probabilities: list[float], n: int) -> None:
    header = f"{'index':>7}{'key':>10}{'probability':>15}"
    print(header)
    print("-" * len(header))
    for i in range(1, n + 1):
        print(f"{i:>7}{keys[i]:>10}{probabilities[i]:>15.6f}")
    print("-" * len(header))
    print(f"{'sum':>17}{sum(probabilities[1:]):>15.6f}")

# Displays the level/depth of each key in the resulting tree
def print_depth_table(
    keys: list[int],
    probabilities: list[float],
    depths: dict[int, int],
    n: int,
) -> None:
    header = f"{'key':>8}{'probability':>15}{'depth':>8}{'prob x depth':>16}"
    print(header)
    print("-" * len(header))
    for i in range(1, n + 1):
        depth = depths[keys[i]]
        print(
            f"{keys[i]:>8}{probabilities[i]:>15.6f}{depth:>8}"
            f"{probabilities[i] * depth:>16.6f}"
        )

# Outputs the experimental comparison between the Optimal BST and conventional BSTs, highlighting expected search cost and average depth
def print_comparison(keys: list[int], probabilities: list[float]) -> None:
    results = compare_trees(keys, probabilities)
    order = ("optimal", "sequential", "balanced")

    for name in order:
        print(f"--- {name} tree ---")
        for line in render_tree(results[name]["tree"]):
            print(line)
        print()

    header = (
        f"{'tree':<12}{'exp. cost':>12}{'avg depth':>12}"
        f"{'height':>8}{'time (s)':>14}"
    )
    print(header)
    print("-" * len(header))
    for name in order:
        result = results[name]
        print(
            f"{name:<12}{result['expected_cost']:>12.4f}"
            f"{result['average_depth']:>12.4f}{result['height']:>8}"
            f"{result['build_seconds']:>14.6f}"
        )
    print()

    optimal_cost = results["optimal"]["expected_cost"]
    for name in ("sequential", "balanced"):
        baseline_cost = results[name]["expected_cost"]
        improvement = (baseline_cost - optimal_cost) / baseline_cost * 100.0
        print(
            f"optimal against {name:<12}{optimal_cost:.4f} vs {baseline_cost:.4f}"
            f"  ->{improvement:>7.2f}% lower expected cost"
        )

# Coordinates the algorithm execution and formats all required outputs for the report submission
def run(args: argparse.Namespace) -> None:
    keys, probabilities = load_input(args)
    n = len(keys) - 1

    dp_start = time.perf_counter()
    cost, root_table = build_dp_tables(probabilities)
    dp_seconds = time.perf_counter() - dp_start

    build_start = time.perf_counter()
    tree = build_tree(keys, probabilities, root_table, 1, n)
    build_seconds = time.perf_counter() - build_start

    print("=== 1. input ===")
    print_input_table(keys, probabilities, n)
    print()

    auto_suppressed = n > MAX_TABLE_KEYS
    if args.no_tables:
        print("=== 2-3. cost and root tables ===")
        print("suppressed by --no-tables")
        print()
    elif auto_suppressed:
        print("=== 2-3. cost and root tables ===")
        print(
            f"suppressed automatically: n = {n} is above {MAX_TABLE_KEYS} and the "
            f"tables no longer fit a terminal (--no-tables controls this too)"
        )
        print()
    else:
        print("=== 2. cost table C[i][j] ===")
        for line in render_upper_triangle(cost, n, 9, True):
            print(line)
        print()

        print("=== 3. root table root[i][j] ===")
        for line in render_upper_triangle(root_table, n, 6, False):
            print(line)
        print()

    table_cost = cost[1][n]
    print("=== 4. minimum expected search cost ===")
    print(f"C[1][{n}] = {table_cost:.6f}")
    print()

    print("=== 5. optimal tree (right subtree above the node, left below) ===")
    for line in render_tree(tree):
        print(line)
    print()

    depths = collect_depths(tree)
    print("=== 6. depth of each key ===")
    print_depth_table(keys, probabilities, depths, n)
    print()

    tree_cost = expected_cost_from_tree(tree)
    difference = abs(tree_cost - table_cost)
    print("=== 7. verification against the reconstructed tree ===")
    print(f"{'cost from the tree':<25}= {tree_cost:.10f}")
    print(f"{f'C[1][{n}] from the table':<25}= {table_cost:.10f}")
    print(f"{'absolute difference':<25}= {difference:.2e}")
    print(f"{'agree within 1e-9':<25}: {'yes' if difference < 1e-9 else 'NO'}")
    print()

    # Records and outputs execution time to support experimental analysis of varying input sizes
    print("=== 8. execution time ===")
    print(f"{'DP table fill':<22}= {dp_seconds:.6f} s")
    print(f"{'tree reconstruction':<22}= {build_seconds:.6f} s")
    print(f"{'total':<22}= {dp_seconds + build_seconds:.6f} s")
    print()

    if args.compare:
        print("=== 9. comparison with conventional binary search trees ===")
        print_comparison(keys, probabilities)

# Ensures invalid inputs (such as probabilities that do not sum to 1) are handled appropriately
def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    try:
        run(args)
    except InvalidInputError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())