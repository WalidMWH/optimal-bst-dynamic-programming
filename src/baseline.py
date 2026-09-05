# Constructs conventional BSTs to compare:
# Expected search cost, average search depth, and execution time against the Optimal BST

import time
from obst import build_dp_tables
from tree import Node, build_tree, collect_depths, expected_cost_from_tree

# Builds a conventional BST through sequential insertion to evaluate worst-case tree structure and depth
def build_sequential_bst(keys: list[int], probabilities: list[float]) -> Node | None:
    root: Node | None = None

    for index in range(1, len(keys)):
        node = Node(keys[index], probabilities[index], 1)
        if root is None:
            root = node
            continue

        # Simulates a degenerate tree structure to contrast with the optimal arrangement of keys
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

# Constructs a height-balanced conventional BST for comparison against the optimal tree
def build_balanced_bst(keys: list[int], probabilities: list[float]) -> Node | None:
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

    # Roots the sub problem at the middle key to ensure a balanced baseline structure
    mid = (i + j) // 2
    return Node(
        key=keys[mid],
        probability=probabilities[mid],
        depth=depth,
        left=_build_balanced(keys, probabilities, i, mid - 1, depth + 1),
        right=_build_balanced(keys, probabilities, mid + 1, j, depth + 1),
    )

# Calculates the average search depth of the constructed tree to fulfill the comparison requirement
def average_depth(node: Node | None) -> float:
    depths = collect_depths(node)
    if not depths:
        return 0.0
    return sum(depths.values()) / len(depths)

# Determines the maximum depth to report on the resulting tree structure
def tree_height(node: Node | None) -> int:
    depths = collect_depths(node)
    if not depths:
        return 0
    return max(depths.values())

# Records execution times and performance metrics to evaluate the algorithms across different inputs
def compare_trees(keys: list[int], probabilities: list[float]) -> dict:
    n = len(keys) - 1

    # Measures the construction execution time of the Dynamic Programming approach
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