# Reconstructs the actual Optimal Binary Search Tree from the DP root table
# Verifies the minimum expected search cost by calculating probability times depth

# Represents a key in the Optimal Binary Search Tree, storing its successful-search probability and depth
class Node:
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

# Recursively reconstructs the tree structure using the root table to find the selected root for each sub problem
def build_tree(
    keys: list[int],
    probabilities: list[float],
    root_table: list[list[int]],
    i: int,
    j: int,
    depth: int = 1,
) -> Node | None:
    # Base case for empty sub problems where expected search cost is 0
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

# Records the level/depth of each key in the resulting Optimal Binary Search Tree
def collect_depths(node: Node | None) -> dict[int, int]:
    if node is None:
        return {}
    depths = {node.key: node.depth}
    depths.update(collect_depths(node.left))
    depths.update(collect_depths(node.right))
    return depths

# Verifies the resulting expected cost using the constructed tree
def expected_cost_from_tree(node: Node | None) -> float:
    # Independent of the DP cost table, verifying the expected search cost from the final tree structure
    if node is None:
        return 0.0
    return (
        node.probability * node.depth
        + expected_cost_from_tree(node.left)
        + expected_cost_from_tree(node.right)
    )

# Displays the structure of the Optimal Binary Search Tree for reporting
def render_tree(node: Node | None) -> list[str]:
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

    # Helper visual logic to format and display the tree structure
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

# Traverses the structure to verify it matches a conventional BST key ordering for comparison
def traversal_inorder(node: Node | None) -> list[int]:
    if node is None:
        return []
    return traversal_inorder(node.left) + [node.key] + traversal_inorder(node.right)