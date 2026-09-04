"""Rebuild the optimal binary search tree from the root table filled by obst.py.

Given root[i][j] for keys i..j, the tree is rebuilt top down: the root of a range
is root[i][j], and the two sides recurse on i..r-1 and r+1..j one level deeper.
The reconstructed tree gives an independent check of the DP result, since summing
probability times depth over the nodes must reproduce C[1][n].
Indexing is 1-based, matching obst.py, and the root sits at depth 1.
"""


class Node:
    """One key of the optimal tree, with its probability, depth and children."""

    def __init__(
        self,
        key: int,
        probability: float,
        depth: int,
        left: "Node | None" = None,
        right: "Node | None" = None,
    ) -> None:
        self.key = key
        self.probability = probability
        self.depth = depth
        self.left = left
        self.right = right


def build_tree(
    keys: list[int],
    probabilities: list[float],
    root_table: list[list[int]],
    i: int,
    j: int,
    depth: int = 1,
) -> Node | None:
    """Rebuild the optimal subtree over keys i..j from the root table."""
    # Empty range: the same case that gives C[i][j] = 0 in the DP.
    if i > j:
        return None

    r = root_table[i][j]
    return Node(
        key=keys[r],
        probability=probabilities[r],
        depth=depth,
        left=build_tree(keys, probabilities, root_table, i, r - 1, depth + 1),
        right=build_tree(keys, probabilities, root_table, r + 1, j, depth + 1),
    )


def collect_depths(node: Node | None) -> dict[int, int]:
    """Map each key in the tree to the depth it ended up at."""
    if node is None:
        return {}

    depths = {node.key: node.depth}
    depths.update(collect_depths(node.left))
    depths.update(collect_depths(node.right))
    return depths


def expected_cost_from_tree(node: Node | None) -> float:
    """Sum probability times depth over the tree."""
    # Deliberately walks the nodes instead of reading cost[1][n]: this is the
    # value that confirms the DP, so it must not come from the DP.
    if node is None:
        return 0.0

    return (
        node.probability * node.depth
        + expected_cost_from_tree(node.left)
        + expected_cost_from_tree(node.right)
    )


def render_tree(node: Node | None) -> list[str]:
    """Draw the tree sideways, right subtree above each node and left below."""
    lines: list[str] = []
    _render_sideways(node, lines, "", None)
    return lines


def _render_sideways(
    node: Node | None,
    lines: list[str],
    prefix: str,
    side: str | None,
) -> None:
    if node is None:
        return

    # A bar is drawn on the side of the node that faces its parent's line, so
    # the branch stays visually connected across the subtree printed in between.
    if side == "right":
        connector = "/-- "
        above_prefix = prefix + "    "
        below_prefix = prefix + "|   "
    elif side == "left":
        connector = "\\-- "
        above_prefix = prefix + "|   "
        below_prefix = prefix + "    "
    else:
        connector = ""
        above_prefix = prefix + "    "
        below_prefix = prefix + "    "

    _render_sideways(node.right, lines, above_prefix, "right")
    lines.append(prefix + connector + str(node.key))
    _render_sideways(node.left, lines, below_prefix, "left")


def traversal_inorder(node: Node | None) -> list[int]:
    """List the keys in inorder, which must come out sorted for a valid BST."""
    if node is None:
        return []

    return traversal_inorder(node.left) + [node.key] + traversal_inorder(node.right)


if __name__ == "__main__":
    # obst.py sits next to this file; both are run directly from inside src/.
    from obst import build_dp_tables

    def _run_case(
        name: str,
        raw_keys: list[int],
        raw_probabilities: list[float],
        target_cost: float,
        target_root_key: int,
    ) -> bool:
        n = len(raw_keys)
        # Pad both lists to the 1-based layout obst.py expects.
        keys = [0] + list(raw_keys)
        probabilities = [0.0] + list(raw_probabilities)

        cost, root_table = build_dp_tables(probabilities)
        tree = build_tree(keys, probabilities, root_table, 1, n)

        depths = collect_depths(tree)
        inorder = traversal_inorder(tree)
        tree_cost = expected_cost_from_tree(tree)
        table_cost = cost[1][n]
        difference = abs(tree_cost - table_cost)

        print(f"=== {name} ===")
        print(f"target: cost {target_cost:.4f}, root key {target_root_key}")
        print()

        print("tree (right subtree above the node, left subtree below):")
        for line in render_tree(tree):
            print(line)
        print()

        probability_of = dict(zip(raw_keys, raw_probabilities))
        print(f"{'key':>6}{'prob':>9}{'depth':>8}{'prob x depth':>15}")
        print("-" * 38)
        for key in inorder:
            probability = probability_of[key]
            depth = depths[key]
            print(
                f"{key:>6}{probability:>9.2f}{depth:>8}"
                f"{probability * depth:>15.4f}"
            )
        print("-" * 38)

        print(f"{'total from tree':<21}= {tree_cost:.4f}")
        print(f"{f'C[1][{n}] from table':<21}= {table_cost:.4f}")
        print(f"{'absolute difference':<21}= {difference:.2e}")
        print()

        print(f"inorder  : {inorder}")
        print(f"expected : {sorted(raw_keys)}")
        print()

        # Summed floats rarely land on the exact target, so compare with slack.
        cost_ok = difference < 1e-9
        order_ok = inorder == sorted(raw_keys)
        # len(inorder) catches a duplicated key, which the dict would hide.
        depths_ok = (
            len(depths) == n
            and len(inorder) == n
            and sorted(depths) == sorted(raw_keys)
        )
        target_ok = (
            tree is not None
            and abs(tree_cost - target_cost) < 1e-9
            and tree.key == target_root_key
        )

        passed = cost_ok and order_ok and depths_ok and target_ok
        print(f"cost match   : {'ok' if cost_ok else 'FAILED'}")
        print(f"inorder      : {'ok' if order_ok else 'FAILED'}")
        print(f"depth map    : {'ok' if depths_ok else 'FAILED'}")
        print(f"target match : {'ok' if target_ok else 'FAILED'}")
        print("RESULT: " + ("PASS" if passed else "FAIL"))
        print()
        return passed

    results = [
        _run_case(
            "Case A",
            raw_keys=[10, 20, 30],
            raw_probabilities=[0.2, 0.5, 0.3],
            target_cost=1.5,
            target_root_key=20,
        ),
        _run_case(
            "Case B",
            raw_keys=[10, 20, 30, 40, 50],
            raw_probabilities=[0.10, 0.20, 0.40, 0.20, 0.10],
            target_cost=1.8,
            target_root_key=30,
        ),
    ]

    print("=== Summary ===")
    for label, ok in zip(("Case A", "Case B"), results):
        print(f"{label}: {'PASS' if ok else 'FAIL'}")
    print(f"Overall: {'PASS' if all(results) else 'FAIL'}")