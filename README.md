# Optimal Binary Search Tree

A dynamic programming implementation that builds the binary search tree
minimizing expected search cost, given sorted keys and their access
probabilities.

## Problem

A binary search tree can be arranged in many ways over the same set of keys.
When some keys are searched more often than others, the arrangement changes how
many comparisons an average search needs.

For keys `k₁ … kₙ` with access probabilities `p₁ … pₙ`, the expected search cost
is

```
E = Σ pᵢ × depth(kᵢ)          (root at depth 1)
```

The probabilities are fixed by the input, so the depths are the only thing under
our control. The task is to find the tree shape that minimizes `E`.

Exhaustive search is not viable: the number of distinct BSTs on `n` keys is the
Catalan number `Cₙ`, which is 42 for `n = 5` and roughly 6.56 billion for
`n = 20`. Dynamic programming solves `n = 20` in 1,540 inner operations.

## Algorithm

**State.** `C[i][j]` is the minimum expected cost of an optimal BST over keys
`kᵢ … kⱼ`, with depth measured relative to that subtree's own root.

**Recurrence.**

```
C[i][j] = min over r in [i..j] of { C[i][r-1] + C[r+1][j] } + W(i, j)

W(i, j) = pᵢ + … + pⱼ
```

**Base case.** `C[i][j] = 0` when `i > j`. The single-key case `C[i][i] = pᵢ`
follows from it.

The `W(i, j)` term appears because choosing `r` as root pushes every key in the
interval one level deeper relative to its position within its own subtree, so
each contributes one extra comparison weighted by its probability.

`W(i, j)` is evaluated in O(1) from a precomputed prefix-sum array. Computing it
inside the innermost loop would make the algorithm O(n⁴).

**Complexity.** Time O(n³), space O(n²). There are Θ(n²) subproblems and each
tries up to O(n) roots; the exact inner-loop count is `n(n+1)(n+2)/6`.

## Results

TBD

## Project structure

```
main.py              CLI entry point
src/                 algorithm modules
tests/               correctness tests
data/                sample inputs
results/             sample outputs, timing data, plots
docs/                project report
```

## Requirements

Python 3.10 or newer. The core program uses only the standard library.

`matplotlib` is needed only to render the complexity plot, and `pytest` only to
run the test suite:

```bash
pip install -r requirements.txt
```

## Usage

TBD

## Input format

TBD

## Testing

TBD

## License

MIT. See [LICENSE](LICENSE).