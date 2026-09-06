# Optimal Binary Search Tree using Dynamic Programming
*CMP3005 - Analysis Of Algorithms - Course Project*

**Done By:** WALID M. W. KOBAHALABI - 2477317  
**Instructor:** Dr. MD IMRAN HOSEN  
**University:** Bahçeşehir Üniversitesi

> **Academic Integrity Notice**  
> This repository is coursework submitted for CMP3005 at Bahçeşehir
> Üniversitesi. It is published for reference and learning. If you are taking
> this or a similar course, do not submit any part of this work as your own,
> doing so is plagiarism under your institution's academic integrity policy.

📄 **[Project Report (PDF)](docs/2477317_CMP3005_OBST_Using_DP_Project_Report.pdf)**

## Project Overview

Design and implement an Optimal Binary Search Tree (OBST) using Dynamic
Programming. Given a set of sorted keys and their corresponding
successful-search probabilities, the program constructs the Binary Search Tree
that minimizes the expected number of comparisons required to search for a key.

The program formulates the problem as a set of overlapping subproblems,
constructs the Dynamic Programming tables, determines the minimum expected
search cost, and reconstructs the resulting optimal binary search tree.

## Dynamic Programming Formulation

Let the sorted keys be K = {k₁, k₂, …, kₙ} and let pᵢ be the successful-search
probability associated with key kᵢ.

C[i,j] = minimum expected search cost for keys kᵢ through kⱼ

```
C[i,j] = min { C[i,k-1] + C[k+1,j] } + Σ ps     for 1 ≤ i ≤ j ≤ n
C[i,i] = pᵢ
```

The summation term is added because when a subtree is placed one level deeper,
every key in that subtree requires one additional comparison.

Standard OBST Dynamic Programming: Time = O(n³), Space = O(n²)

## Requirements

Python 3.10 or newer. The core program uses only the standard library.

`matplotlib` is needed only for the complexity plot and `pytest` only for the
test suite:

```
pip install -r requirements.txt
```

## How to Run

The program is run from the project root directory. With no arguments it uses
the example input from the project description:

```
python main.py
```

Other options:

| Command | Description |
|---|---|
| `python main.py --file data/example5.txt` | Read the input from a file |
| `python main.py --interactive` | Enter the input manually |
| `python main.py --random 20` | Generate 20 random keys |
| `python main.py --compare` | Also compare with a conventional BST |
| `python main.py --no-tables` | Hide the cost and root tables |

To run the experimental analysis for multiple input sizes:

```
python experiments.py
```

This writes `results/timings.csv` and `results/complexity.png`.

## How to Provide Input

The program accepts the number of keys n, the n sorted keys, and the
successful-search probability for each key. The probabilities must be
non-negative and must sum to 1, allowing for minor floating-point rounding.

Input files are plain text. Lines beginning with `#` and blank lines are
ignored. The first line is the number of keys, and each following line holds a
key and its probability:

```
# Example input from the project description
5
10 0.10
20 0.20
30 0.40
40 0.20
50 0.10
```

Invalid input is handled with an error message instead of a crash:

```
$ python main.py --file data/invalid_sum.txt
Error: probabilities must sum to 1, they sum to 0.9 (tolerance 1e-06)
```

The `data/` folder contains three valid sample inputs and four invalid ones.

## Sample Output

Running `python main.py` with the example input:

```
=== 1. input ===
  index       key    probability
--------------------------------
      1        10       0.100000
      2        20       0.200000
      3        30       0.400000
      4        40       0.200000
      5        50       0.100000
--------------------------------
              sum       1.000000

=== 2. cost table C[i][j] ===
              1        2        3        4        5
      ---------------------------------------------
   1 |   0.1000   0.4000   1.1000   1.5000   1.8000
   2 |            0.2000   0.8000   1.2000   1.5000
   3 |                     0.4000   0.8000   1.1000
   4 |                              0.2000   0.4000
   5 |                                       0.1000

=== 3. root table root[i][j] ===
           1     2     3     4     5
      ------------------------------
   1 |     1     2     3     3     3
   2 |           2     3     3     3
   3 |                 3     3     3
   4 |                       4     4
   5 |                             5

=== 4. minimum expected search cost ===
C[1][5] = 1.800000

=== 5. optimal tree (right subtree above the node, left below) ===
        /-- 50
    /-- 40
30
    \-- 20
        \-- 10

=== 6. depth of each key ===
     key    probability   depth    prob x depth
-----------------------------------------------
      10       0.100000       3        0.300000
      20       0.200000       2        0.400000
      30       0.400000       1        0.400000
      40       0.200000       2        0.400000
      50       0.100000       3        0.300000

=== 7. verification against the reconstructed tree ===
cost from the tree       = 1.8000000000
C[1][5] from the table   = 1.8000000000
absolute difference      = 2.22e-16
agree within 1e-9        : yes

=== 8. execution time ===
DP table fill         = 0.000014 s
tree reconstruction   = 0.000008 s
total                 = 0.000021 s
```

## Comparison with a Conventional BST

Running `python main.py --compare` also builds two conventional trees over the
same keys:

| Tree | Expected cost | Average depth | Height |
|---|---|---|---|
| Sequential insertion | 3.0000 | 3.0000 | 5 |
| Balanced | 1.9000 | 2.2000 | 3 |
| Optimal BST | 1.8000 | 2.2000 | 3 |

## Experimental Results

Running `python experiments.py` times the DP table fill over several input
sizes, averaged across 5 random instances per size:

| Number of Keys (n) | Execution Time (s) | Inner Loop Count |
|---:|---:|---:|
| 5 | 0.000008 | 35 |
| 10 | 0.000024 | 220 |
| 20 | 0.000113 | 1,540 |
| 50 | 0.001168 | 22,100 |
| 100 | 0.008437 | 171,700 |
| 200 | 0.061855 | 1,353,400 |
| 400 | 0.550207 | 10,746,800 |

The inner loop count is n(n+1)(n+2)/6, the exact number of candidate roots the
algorithm evaluates. Doubling n from 200 to 400 multiplied the execution time
by 8.9, close to the factor of 8 that cubic growth predicts.

![Execution time against n](results/complexity.png)

The measured times are plotted on logarithmic axes against a reference line
proportional to n³. The two curves run together for the larger input sizes,
while the smallest sizes sit above the reference because their execution time
is dominated by fixed overhead rather than by the algorithm itself. The results
are discussed further in the report.

## Testing

```
pytest tests/
```

185 tests covering the DP tables, the tree reconstruction, the input
validation, and the conventional BST baselines.

## Project Structure

```
main.py              program entry point
experiments.py       execution time experiments
src/obst.py          DP cost table and root table
src/tree.py          tree reconstruction and cost verification
src/baseline.py      conventional BSTs for comparison
src/validation.py    input reading and validation
tests/               test suite
data/                sample input files
results/             experimental results and sample output
docs/                project report (PDF)
```

## License

MIT. See [LICENSE](LICENSE).