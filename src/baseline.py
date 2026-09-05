"""Conventional binary search trees, built for comparison against the optimal one.

Both baselines ignore the probabilities: sequential insertion, which degenerates
into a right chain for sorted keys, and the middle-key split, which is height
balanced. Each tree is scored by expected cost, average depth, height and build
time, so the gain the DP buys over ordinary construction is visible.
"""

import time

# obst.py and tree.py sit next to this file; all three are run from inside src/.
from obst import build_dp_tables
from tree import (
    Node,
    build_tree,
    collect_depths,
    expected_cost_from_tree,
    render_tree,
    traversal_inorder,
)


def build_sequential_bst(keys: list[int], probabilities: list[float]) -> Node | None:
    """Insert the keys in the order given using ordinary BST insertion, no balancing."""
    root: Node | None = None

    for index in range(1, len(keys)):
        node = Node(keys[index], probabilities[index], 1)
        if root is None:
            root = node
            continue

        # Sorted input sends every key down the right spine, which is exactly the
        # degenerate worst case this baseline is meant to show.
        current = root
        depth = 1
        while True:
            depth += 1
            if node.key < current.key:
                if current.left is None:
                    node.depth = depth
                    current.left = node
                    break
                current = current.left
            else:
                if current.right is None:
                    node.depth = depth
                    current.right = node
                    break
                current = current.right

    return root


def build_balanced_bst(keys: list[int], probabilities: list[float]) -> Node | None:
    """Build the height-balanced tree that roots every range at its middle key."""
    return _build_balanced(keys, probabilities, 1, len(keys) - 1, 1)


def _build_balanced(
    keys: list[int],
    probabilities: list[float],
    i: int,
    j: int,
    depth: int,
) -> Node | None:
    if i > j:
        return None

    # Floor division takes the lower middle, so even ranges split the same way
    # on every run.
    mid = (i + j) // 2
    return Node(
        key=keys[mid],
        probability=probabilities[mid],
        depth=depth,
        left=_build_balanced(keys, probabilities, i, mid - 1, depth + 1),
        right=_build_balanced(keys, probabilities, mid + 1, j, depth + 1),
    )


def average_depth(node: Node | None) -> float:
    """Mean depth over the nodes, counting every key equally rather than by probability."""
    depths = collect_depths(node)
    if not depths:
        return 0.0

    return sum(depths.values()) / len(depths)


def tree_height(node: Node | None) -> int:
    """Depth of the deepest node, or 0 for an empty tree."""
    depths = collect_depths(node)
    if not depths:
        return 0

    return max(depths.values())


def compare_trees(keys: list[int], probabilities: list[float]) -> dict:
    """Build the optimal, sequential and balanced trees and measure each of them."""
    n = len(keys) - 1

    # The DP fill is part of what an optimal tree costs to produce, so it is timed
    # together with the reconstruction; the baselines have no such precomputation.
    start = time.perf_counter()
    _, root_table = build_dp_tables(probabilities)
    optimal = build_tree(keys, probabilities, root_table, 1, n)
    optimal_seconds = time.perf_counter() - start

    start = time.perf_counter()
    sequential = build_sequential_bst(keys, probabilities)
    sequential_seconds = time.perf_counter() - start

    start = time.perf_counter()
    balanced = build_balanced_bst(keys, probabilities)
    balanced_seconds = time.perf_counter() - start

    return {
        name: {
            "tree": tree,
            "expected_cost": expected_cost_from_tree(tree),
            "average_depth": average_depth(tree),
            "height": tree_height(tree),
            "build_seconds": seconds,
        }
        for name, tree, seconds in (
            ("optimal", optimal, optimal_seconds),
            ("sequential", sequential, sequential_seconds),
            ("balanced", balanced, balanced_seconds),
        )
    }


if __name__ == "__main__":
    raw_keys = [10, 20, 30, 40, 50]
    raw_probabilities = [0.10, 0.20, 0.40, 0.20, 0.10]

    # Pad to the 1-based layout obst.py and tree.py expect.
    keys = [0] + raw_keys
    probabilities = [0.0] + raw_probabilities

    results = compare_trees(keys, probabilities)
    order = ("optimal", "sequential", "balanced")

    print("keys  :  " + "  ".join(f"{k:>6}" for k in raw_keys))
    print("prob  :  " + "  ".join(f"{p:>6.2f}" for p in raw_probabilities))
    print()

    for name in order:
        print(f"=== {name} tree (right subtree above the node, left below) ===")
        for line in render_tree(results[name]["tree"]):
            print(line)
        print(f"inorder: {traversal_inorder(results[name]['tree'])}")
        print()

    print("=== comparison ===")
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
    print("=== improvement of the optimal tree ===")
    for name in ("sequential", "balanced"):
        baseline_cost = results[name]["expected_cost"]
        improvement = (baseline_cost - optimal_cost) / baseline_cost * 100.0
        print(
            f"vs {name:<12}{optimal_cost:.4f} against {baseline_cost:.4f}"
            f"  ->{improvement:>7.2f}% lower expected cost"
        )
    print()

    targets = {
        "optimal": (1.80, 3),
        "sequential": (3.00, 5),
        "balanced": (1.90, 3),
    }

    print("=== verification ===")
    checks = []
    for name in order:
        target_cost, target_height = targets[name]
        result = results[name]
        # Summed floats rarely land on the exact target, so compare with slack.
        cost_ok = abs(result["expected_cost"] - target_cost) < 1e-9
        height_ok = result["height"] == target_height
        checks.append(cost_ok and height_ok)
        print(
            f"{name:<12}cost {result['expected_cost']:.4f} vs {target_cost:.2f} "
            f"{'ok' if cost_ok else 'FAILED':<8}"
            f"height {result['height']} vs {target_height} "
            f"{'ok' if height_ok else 'FAILED'}"
        )
    print()
    print("RESULT: " + ("PASS" if all(checks) else "FAIL"))