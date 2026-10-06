# OR-Library Capacitated Warehouse Location instances

Source: J. E. Beasley, OR-Library
- Info page: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html
- Files:     https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/

The files are **unmodified** copies of the originals. `SHA256SUMS` holds the hashes
of the official Brunel files; check them with `python scripts/verify_data.py`
(or `sha256sum -c SHA256SUMS` on Linux/macOS).

| Category | Instance | Facilities (m) | Customers (n) |
|----------|----------|----------------|---------------|
| Small    | cap41, cap42   | 16 | 50 |
| Medium   | cap101, cap102 | 25 | 50 |
| Large    | cap121, cap122 | 50 | 50 |

`capopt.txt` lists the known optimal *single-objective* totals (f1 + f2). Use it only as a
sanity reference, not as an input.

## File format
```
m n
capacity_1  fixed_cost_1        <- m lines, one per facility (S_i, F_i)
...
demand_1                         <- for each customer j: d_j ...
c_1j c_2j ... c_mj               <- ... followed by m allocation costs C_ij (may wrap over lines)
...
```
`C_ij` is the cost of allocating **all** of customer j's demand to facility i. It must
NOT be multiplied by `d_j` again (Objective 2 = sum C_ij x_ij).

## Oversized customers in cap41 / cap42
In cap41 and cap42 every facility has capacity 5000, but two customers have demand
5495 and 12912. They fit in no single facility, so strict single-sourcing
(each customer assigned to exactly one facility) has no feasible solution.

**Group decision:** a customer whose demand exceeds the largest facility capacity
may be split across several open facilities. All other customers are single-sourced.
Capacity is never exceeded and the data is not changed.

- For a split customer, facility i is charged `C_ij * (q_ij / d_j)`, where `q_ij` is
  the amount it serves (shares sum to `d_j`). Unsplit customers are charged `C_ij`.
- Only cap41 and cap42 are affected; cap101, cap102, cap121 and cap122 use strict
  single-sourcing.
- The same rule is used by both MOEAs.
