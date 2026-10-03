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
