"""Pytest suite for the optimal binary search tree implementation.

Hand-computed golden cases pin the DP down exactly, randomised property checks
and an exponential brute force confirm it on inputs nobody worked out by hand,
and the rest covers validation, parsing and the conventional-tree baselines.
"""

import sys
from pathlib import Path

import pytest

# tests/ is a sibling of src/, so src/ has to be on the path before the modules
# under test can be imported by their plain names.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baseline import (  # noqa: E402
    average_depth,
    build_balanced_bst,
    build_sequential_bst,
    compare_trees,
    tree_height,
)
from obst import (  # noqa: E402
    build_dp_tables,
    compute_prefix_sums,
    interval_probability,
)
from tree import (  # noqa: E402
    Node,
    build_tree,
    collect_depths,
    expected_cost_from_tree,
    traversal_inorder,
)
from validation import (  # noqa: E402
    InvalidInputError,
    generate_random,
    parse_lines,
    validate,
)

EXAMPLE_KEYS = [10, 20, 30, 40, 50]
EXAMPLE_PROBABILITIES = [0.10, 0.20, 0.40, 0.20, 0.10]

# Keys, probabilities, optimal cost, optimal root index, depth of every key.
GOLDEN_CASES = [
    ([10, 20, 30], [0.2, 0.5, 0.3], 1.5, 2, {20: 1, 10: 2, 30: 2}),
    (
        EXAMPLE_KEYS,
        EXAMPLE_PROBABILITIES,
        1.8,
        3,
        {30: 1, 20: 2, 40: 2, 10: 3, 50: 3},
    ),
]
GOLDEN_IDS = ["three keys", "five keys"]

# Each row of the 5-key tables, starting at the diagonal.
EXPECTED_COST_ROWS = {
    1: [0.1, 0.4, 1.1, 1.5, 1.8],
    2: [0.2, 0.8, 1.2, 1.5],
    3: [0.4, 0.8, 1.1],
    4: [0.2, 0.4],
    5: [0.1],
}
EXPECTED_ROOT_ROWS = {
    1: [1, 2, 3, 3, 3],
    2: [2, 3, 3, 3],
    3: [3, 3, 3],
    4: [4, 4],
    5: [5],
}

RANDOM_INSTANCES = [
    (n, seed) for n in (1, 2, 3, 5, 8, 13, 21, 30) for seed in (0, 1, 7)
]


def padded(
    keys: list[int], probabilities: list[float]
) -> tuple[list[int], list[float]]:
    """Add the unused index 0 padding that every module expects."""
    return [0] + keys, [0.0] + probabilities


def optimal_tree(
    keys: list[int], probabilities: list[float]
) -> tuple[list[list[float]], list[list[int]], Node | None]:
    """Run the DP and rebuild the tree from already padded lists."""
    n = len(keys) - 1
    cost, root_table = build_dp_tables(probabilities)
    return cost, root_table, build_tree(keys, probabilities, root_table, 1, n)


def brute_force_cost(probabilities: list[float], i: int, j: int) -> float:
    """Minimum expected cost over every BST shape on keys i..j, with no memoisation."""
    if i > j:
        return 0.0

    weight = sum(probabilities[i : j + 1])
    return weight + min(
        brute_force_cost(probabilities, i, r - 1)
        + brute_force_cost(probabilities, r + 1, j)
        for r in range(i, j + 1)
    )


# --- Group 1: golden cases verified by hand ---


@pytest.mark.parametrize(
    "raw_keys, raw_probabilities, expected_cost, expected_root, expected_depths",
    GOLDEN_CASES,
    ids=GOLDEN_IDS,
)
def test_golden_optimal_cost(
    raw_keys, raw_probabilities, expected_cost, expected_root, expected_depths
):
    """C[1][n] equals the cost worked out by hand for each golden case."""
    keys, probabilities = padded(raw_keys, raw_probabilities)
    cost, _ = build_dp_tables(probabilities)
    assert cost[1][len(raw_keys)] == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    "raw_keys, raw_probabilities, expected_cost, expected_root, expected_depths",
    GOLDEN_CASES,
    ids=GOLDEN_IDS,
)
def test_golden_root_index(
    raw_keys, raw_probabilities, expected_cost, expected_root, expected_depths
):
    """The whole-range root index is the one the hand calculation picks."""
    keys, probabilities = padded(raw_keys, raw_probabilities)
    _, root_table = build_dp_tables(probabilities)
    assert root_table[1][len(raw_keys)] == expected_root


@pytest.mark.parametrize(
    "raw_keys, raw_probabilities, expected_cost, expected_root, expected_depths",
    GOLDEN_CASES,
    ids=GOLDEN_IDS,
)
def test_golden_depths(
    raw_keys, raw_probabilities, expected_cost, expected_root, expected_depths
):
    """Every key in the reconstructed tree sits at the depth worked out by hand."""
    keys, probabilities = padded(raw_keys, raw_probabilities)
    _, _, tree = optimal_tree(keys, probabilities)
    assert collect_depths(tree) == expected_depths


@pytest.mark.parametrize("i", sorted(EXPECTED_COST_ROWS))
def test_five_key_cost_table_row(i):
    """Every upper-triangle cost cell of the 5-key example matches the hand table."""
    _, probabilities = padded(EXAMPLE_KEYS, EXAMPLE_PROBABILITIES)
    cost, _ = build_dp_tables(probabilities)
    row = [cost[i][j] for j in range(i, len(EXAMPLE_KEYS) + 1)]
    assert row == pytest.approx(EXPECTED_COST_ROWS[i])


@pytest.mark.parametrize("i", sorted(EXPECTED_ROOT_ROWS))
def test_five_key_root_table_row(i):
    """Every upper-triangle root cell of the 5-key example matches the hand table."""
    _, probabilities = padded(EXAMPLE_KEYS, EXAMPLE_PROBABILITIES)
    _, root_table = build_dp_tables(probabilities)
    row = [root_table[i][j] for j in range(i, len(EXAMPLE_KEYS) + 1)]
    assert row == EXPECTED_ROOT_ROWS[i]


# --- Group 2: single key and other edges ---


def test_single_key_costs_its_own_probability():
    """One key gives a one-node tree at depth 1 costing exactly its probability."""
    keys, probabilities = padded([10], [1.0])
    cost, root_table, tree = optimal_tree(keys, probabilities)

    assert cost[1][1] == pytest.approx(probabilities[1])
    assert root_table[1][1] == 1
    assert tree is not None
    assert (tree.key, tree.depth, tree.left, tree.right) == (10, 1, None, None)


def test_uniform_probabilities_match_brute_force():
    """With no probability to exploit, the DP still finds the true minimum."""
    keys, probabilities = padded([10, 20, 30, 40], [0.25, 0.25, 0.25, 0.25])
    cost, _ = build_dp_tables(probabilities)
    assert cost[1][4] == pytest.approx(brute_force_cost(probabilities, 1, 4))


def test_all_probability_on_one_key_makes_it_the_root():
    """A key of probability 1 must sit at depth 1, since any other tree costs more."""
    keys, probabilities = padded([10, 20, 30, 40, 50], [0.0, 0.0, 0.0, 1.0, 0.0])
    cost, root_table, tree = optimal_tree(keys, probabilities)

    assert root_table[1][5] == 4
    assert tree is not None and tree.key == 40
    assert cost[1][5] == pytest.approx(1.0)


# --- Group 3: prefix sums ---


def test_prefix_sums_are_monotone_and_end_at_the_total():
    """Prefix sums never decrease for non-negative input and finish at the sum."""
    _, probabilities = padded(EXAMPLE_KEYS, EXAMPLE_PROBABILITIES)
    prefix = compute_prefix_sums(probabilities)

    assert prefix[0] == 0.0
    assert all(prefix[j] >= prefix[j - 1] for j in range(1, len(prefix)))
    assert prefix[-1] == pytest.approx(sum(EXAMPLE_PROBABILITIES))


def test_interval_probability_matches_a_direct_sum():
    """W(i, j) from the prefix sums equals the plain slice sum for every range."""
    _, probabilities = padded(EXAMPLE_KEYS, EXAMPLE_PROBABILITIES)
    prefix = compute_prefix_sums(probabilities)
    n = len(EXAMPLE_KEYS)

    for i in range(1, n + 1):
        for j in range(i, n + 1):
            assert interval_probability(prefix, i, j) == pytest.approx(
                sum(probabilities[i : j + 1])
            )


def test_interval_probability_of_an_empty_range_is_zero():
    """An inverted range carries no weight, which is what the DP's base case needs."""
    _, probabilities = padded(EXAMPLE_KEYS, EXAMPLE_PROBABILITIES)
    prefix = compute_prefix_sums(probabilities)

    assert interval_probability(prefix, 3, 2) == 0.0
    assert interval_probability(prefix, 1, 0) == 0.0


# --- Group 4: properties that must hold for any input ---


@pytest.mark.parametrize("n, seed", RANDOM_INSTANCES)
def test_tree_cost_matches_the_table(n, seed):
    """Walking the rebuilt tree reproduces C[1][n] on random instances."""
    keys, probabilities = generate_random(n, seed)
    cost, _, tree = optimal_tree(keys, probabilities)
    assert expected_cost_from_tree(tree) == pytest.approx(cost[1][n])


@pytest.mark.parametrize("n, seed", RANDOM_INSTANCES)
def test_inorder_is_the_sorted_keys(n, seed):
    """The reconstruction is a valid BST, so its inorder walk is the input order."""
    keys, probabilities = generate_random(n, seed)
    _, _, tree = optimal_tree(keys, probabilities)
    assert traversal_inorder(tree) == keys[1:]


@pytest.mark.parametrize("n, seed", RANDOM_INSTANCES)
def test_depth_map_covers_every_key_once(n, seed):
    """Every key appears in the tree exactly once, none lost and none duplicated."""
    keys, probabilities = generate_random(n, seed)
    _, _, tree = optimal_tree(keys, probabilities)
    depths = collect_depths(tree)

    assert len(depths) == n
    assert sorted(depths) == keys[1:]


@pytest.mark.parametrize("n, seed", RANDOM_INSTANCES)
def test_optimal_is_no_worse_than_either_baseline(n, seed):
    """The DP tree never costs more than the sequential or the balanced tree."""
    keys, probabilities = generate_random(n, seed)
    cost, _, _ = optimal_tree(keys, probabilities)

    sequential = expected_cost_from_tree(build_sequential_bst(keys, probabilities))
    balanced = expected_cost_from_tree(build_balanced_bst(keys, probabilities))

    assert cost[1][n] <= sequential + 1e-9
    assert cost[1][n] <= balanced + 1e-9


@pytest.mark.parametrize("n, seed", RANDOM_INSTANCES)
def test_depths_stay_between_one_and_n(n, seed):
    """No key sits above the root or below the depth a degenerate chain would give."""
    keys, probabilities = generate_random(n, seed)
    _, _, tree = optimal_tree(keys, probabilities)
    assert all(1 <= depth <= n for depth in collect_depths(tree).values())


# --- Group 5: brute force cross-check ---


# brute_force_cost re-explores every tree shape instead of memoising, so its work
# grows with the Catalan numbers; n = 7 runs in a moment and n = 12 would not,
# which is why the cross-check stops here.
@pytest.mark.parametrize("n", range(1, 8))
@pytest.mark.parametrize("seed", [0, 3])
def test_dp_matches_exhaustive_search(n, seed):
    """On small inputs the DP finds the same minimum as enumerating every tree."""
    keys, probabilities = generate_random(n, seed)
    cost, _ = build_dp_tables(probabilities)
    assert cost[1][n] == pytest.approx(brute_force_cost(probabilities, 1, n))


# --- Group 6: validation ---


@pytest.mark.parametrize(
    "keys, probabilities, message",
    [
        ([0, 10, 20], [0.0, 0.4, 0.5], "sum"),
        ([0, 10, 20], [0.0, 0.6, 0.5], "sum"),
        ([0, 10, 20], [0.0, -0.1, 1.1], "negative"),
        ([0, 30, 20], [0.0, 0.5, 0.5], "increase"),
        ([0, 10, 10], [0.0, 0.5, 0.5], "distinct"),
        ([0], [0.0], "at least one key"),
        ([0, 10, 20, 30], [0.0, 0.5, 0.5], "differ in length"),
    ],
    ids=[
        "sum below one",
        "sum above one",
        "negative probability",
        "keys out of order",
        "duplicate keys",
        "no keys",
        "length mismatch",
    ],
)
def test_validate_rejects_bad_input(keys, probabilities, message):
    """Each kind of malformed input raises, with a message naming the fault."""
    with pytest.raises(InvalidInputError, match=message):
        validate(keys, probabilities)


def test_validate_accepts_a_sum_inside_the_tolerance():
    """Rounding in the input is tolerated: a sum off by 1e-9 is still valid."""
    keys, probabilities = padded([10, 20], [0.5, 0.5 + 1e-9])
    assert validate(keys, probabilities) is None


# --- Group 7: parsing the text input format ---


def test_parse_lines_ignores_comments_and_blanks():
    """Comment and blank lines are skipped wherever they appear in the file."""
    lines = [
        "# the assignment's small example\n",
        "\n",
        "3\n",
        "10 0.2\n",
        "   # a comment between the data lines\n",
        "20 0.5\n",
        "\n",
        "30 0.3\n",
    ]
    keys, probabilities = parse_lines(lines)

    assert keys == [0, 10, 20, 30]
    assert probabilities == pytest.approx([0.0, 0.2, 0.5, 0.3])


def test_parse_lines_rejects_a_missing_count():
    """A file holding nothing but comments has no key count to read."""
    with pytest.raises(InvalidInputError, match="empty"):
        parse_lines(["# nothing here\n", "\n"])


def test_parse_lines_rejects_a_non_integer_count():
    """The first surviving line must be the number of keys."""
    with pytest.raises(InvalidInputError, match="not an integer"):
        parse_lines(["three\n", "10 0.5\n"])


def test_parse_lines_rejects_too_few_data_lines():
    """A count larger than the number of key lines is caught before the DP runs."""
    with pytest.raises(InvalidInputError, match="key lines"):
        parse_lines(["3\n", "10 0.5\n", "20 0.5\n"])


def test_parse_lines_reports_the_line_number_of_a_bad_field_count():
    """A line with three fields is rejected and the message points at that line."""
    with pytest.raises(InvalidInputError, match="line 3"):
        parse_lines(["2\n", "10 0.5\n", "20 0.5 0.1\n"])


def test_parse_lines_rejects_a_non_numeric_probability():
    """A probability that will not parse as a float is an input error, not a crash."""
    with pytest.raises(InvalidInputError, match="probability is not a number"):
        parse_lines(["2\n", "10 0.5\n", "20 abc\n"])


# --- Group 8: conventional baselines ---


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_sequential_tree_is_a_right_chain(n):
    """Sorted input degenerates into a right chain of height n and mean depth (n+1)/2."""
    keys, probabilities = generate_random(n, seed=1)
    tree = build_sequential_bst(keys, probabilities)

    node = tree
    for expected_key in keys[1:]:
        assert node is not None
        assert node.key == expected_key
        assert node.left is None
        node = node.right
    assert node is None

    assert tree_height(tree) == n
    assert average_depth(tree) == pytest.approx((n + 1) / 2)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 15, 16, 31])
def test_balanced_tree_height(n):
    """The middle-key split gives the minimum possible height for n keys."""
    keys, probabilities = generate_random(n, seed=2)
    tree = build_balanced_bst(keys, probabilities)
    # n.bit_length() is floor(log2 n) + 1 for n >= 1, without any float rounding.
    assert tree_height(tree) == n.bit_length()


def test_five_key_example_comparison():
    """The three trees score exactly as the assignment's worked example says."""
    keys, probabilities = padded(EXAMPLE_KEYS, EXAMPLE_PROBABILITIES)
    results = compare_trees(keys, probabilities)

    assert results["sequential"]["expected_cost"] == pytest.approx(3.0)
    assert results["balanced"]["expected_cost"] == pytest.approx(1.9)
    assert results["optimal"]["expected_cost"] == pytest.approx(1.8)
    assert results["balanced"]["average_depth"] == pytest.approx(2.2)
    assert results["optimal"]["average_depth"] == pytest.approx(2.2)