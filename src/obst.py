# Implements the Optimal Binary Search Tree using Dynamic Programming
# Formulates the problem to find the minimum expected search cost
# Standard OBST Dynamic Programming: Time O(n^3), Space O(n^2)

# Calculates prefix sums to efficiently compute the summation term of search probabilities
def compute_prefix_sums(probabilities: list[float]) -> list[float]:
    n = len(probabilities) - 1
    prefix: list[float] = [0.0] * (n + 1)
    for j in range(1, n + 1):
        prefix[j] = prefix[j - 1] + probabilities[j]
    return prefix

# Computes the total successful-search probability for keys i through j
def interval_probability(prefix: list[float], i: int, j: int) -> float:
    if i > j:
        return 0.0
    return prefix[j] - prefix[i - 1]

# Constructs the DP cost table C[i,j] and the root table to record the selected root for each sub problem
def build_dp_tables(
    probabilities: list[float],
) -> tuple[list[list[float]], list[list[int]]]:
    n = len(probabilities) - 1
    prefix = compute_prefix_sums(probabilities)

    # Initialize base cases for the DP tables
    # Empty intervals remain 0.0
    cost: list[list[float]] = [[0.0] * (n + 2) for _ in range(n + 2)]
    root: list[list[int]] = [[0] * (n + 2) for _ in range(n + 2)]

    # Formulate the problem as a set of overlapping sub problems based on interval length
    for length in range(1, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1

            # Calculate the summation term for the successful-search probabilities
            weight = interval_probability(prefix, i, j)
            best_cost = float("inf")
            best_root = i

            # Evaluate every possible root k in the interval to minimize the search cost
            for r in range(i, j + 1):
                # Recurrence relation applying the minimum expected search cost of subtrees
                subtree_cost = cost[i][r - 1] + cost[r + 1][j]
                if subtree_cost < best_cost:
                    best_cost = subtree_cost
                    best_root = r

            # Update the DP state C[i,j] and record the optimal root
            cost[i][j] = best_cost + weight
            root[i][j] = best_root

    return cost, root