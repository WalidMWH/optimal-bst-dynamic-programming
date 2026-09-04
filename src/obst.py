"""Optimal binary search tree: the DP stage.

Given sorted keys with search probabilities p_1..p_n (no dummy keys), find the
minimal expected search cost, counting the root at depth 1.
    C[i][j] = min over r in [i..j] of { C[i][r-1] + C[r+1][j] } + W(i, j)
where W(i, j) = p_i + ... + p_j, and C[i][j] = 0 when i > j.
Time O(n^3), space O(n^2). Reconstruction, validation, and the CLI live elsewhere.
"""


def compute_prefix_sums(probabilities: list[float]) -> list[float]:
    """Prefix sums of a 1-based probability list, with S[0] = 0.0."""
    n = len(probabilities) - 1
    prefix: list[float] = [0.0] * (n + 1)
    for j in range(1, n + 1):
        prefix[j] = prefix[j - 1] + probabilities[j]
    return prefix


def interval_probability(prefix: list[float], i: int, j: int) -> float:
    """Total probability of keys i..j in O(1), or 0.0 if the interval is empty."""
    if i > j:
        return 0.0
    return prefix[j] - prefix[i - 1]


def build_dp_tables(
    probabilities: list[float],
) -> tuple[list[list[float]], list[list[int]]]:
    """Fill and return the cost and root tables for keys 1..n."""
    n = len(probabilities) - 1
    prefix = compute_prefix_sums(probabilities)

    # One spare row and column at each end: the recurrence reads cost[i][r-1]
    # (column 0 when r = i = 1) and cost[r+1][j] (row n+1 when r = j = n).
    # Those slots stay 0.0, which is exactly the empty-interval cost.
    cost: list[list[float]] = [[0.0] * (n + 2) for _ in range(n + 2)]
    root: list[list[int]] = [[0] * (n + 2) for _ in range(n + 2)]

    # Shortest intervals first: C[i][j] only ever looks at strictly shorter
    # ranges, so ordering by length guarantees they are already final.
    for length in range(1, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1

            # W(i, j) is the same for every candidate root, so it is looked up
            # once here. Recomputing it inside the r-loop would cost O(n^4).
            weight = interval_probability(prefix, i, j)

            best_cost = float("inf")
            best_root = i

            # Ascending r with a strict <: equal-cost roots leave best_root at
            # the smallest index, so tie-breaking is deterministic.
            for r in range(i, j + 1):
                subtree_cost = cost[i][r - 1] + cost[r + 1][j]
                if subtree_cost < best_cost:
                    best_cost = subtree_cost
                    best_root = r

            cost[i][j] = best_cost + weight
            root[i][j] = best_root

    return cost, root


if __name__ == "__main__":

    def _render_upper_triangle(
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
            lines.append(f"{i:>4} |" + "".join(cells) + "  ")
        return lines

    def _run_case(
        name: str,
        keys: list[int],
        raw_probabilities: list[float],
        expected_cost: float,
        expected_root: int,
    ) -> bool:
        n = len(raw_probabilities)
        # Pad to the 1-based layout the DP functions expect.
        probabilities = [0.0] + list(raw_probabilities)
        cost, root = build_dp_tables(probabilities)

        print(f"=== {name} ===")
        print("index :  " + "  ".join(f"{i:>6}" for i in range(1, n + 1)))
        print("key   :  " + "  ".join(f"{k:>6}" for k in keys))
        print("prob  :  " + "  ".join(f"{p:>6.2f}" for p in raw_probabilities))
        print()

        print("cost table C[i][j] (upper triangle):")
        for line in _render_upper_triangle(cost, n, 9, True):
            print(line)
        print()

        print("root table root[i][j] (upper triangle):")
        for line in _render_upper_triangle(root, n, 6, False):
            print(line)
        print()

        final_cost = cost[1][n]
        final_root = root[1][n]
        print(f"C[1][{n}]    = {final_cost:.4f}   (expected {expected_cost:.4f})")
        print(f"root[1][{n}] = {final_root}        (expected {expected_root})")

        # Summed floats rarely land on the exact target, so compare with slack.
        cost_ok = abs(final_cost - expected_cost) < 1e-9
        root_ok = final_root == expected_root
        passed = cost_ok and root_ok
        print("RESULT: " + ("PASS" if passed else "FAIL"))
        print()
        return passed

    results = [
        _run_case(
            "Case A",
            keys=[10, 20, 30],
            raw_probabilities=[0.2, 0.5, 0.3],
            expected_cost=1.5,
            expected_root=2,
        ),
        _run_case(
            "Case B",
            keys=[10, 20, 30, 40, 50],
            raw_probabilities=[0.10, 0.20, 0.40, 0.20, 0.10],
            expected_cost=1.8,
            expected_root=3,
        ),
    ]

    print("=== Summary ===")
    for label, ok in zip(("Case A", "Case B"), results):
        print(f"{label}: {'PASS' if ok else 'FAIL'}")
    print(f"Overall: {'PASS' if all(results) else 'FAIL'}")