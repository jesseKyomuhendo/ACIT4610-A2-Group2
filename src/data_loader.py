"""Data loading for the OR-Library capacitated warehouse location instances.

This module does three things:
  1. verify_data_integrity()  - checks the data files are unmodified (SHA-256)
  2. load_instance(name)      - reads one cap file into numpy arrays
  3. prepare_instance(inst)   - checks the instance can have a feasible solution
                                and handles customers that are too large for
                                every facility (see OVERSIZED_DEMAND in config.yaml)

Other modules normally only call get_instance(name), which does 2 and 3.

File format (all values separated by whitespace, line breaks have no meaning):
    m n                              number of facilities, number of customers
    capacity fixed_cost              repeated m times (one per facility)
    demand cost_1 ... cost_m         repeated n times (one per customer)
cost_i is the cost of serving ALL of that customer's demand from facility i.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from src.util.fetch_config import CHECKSUM_FILE, DATA_DIR, INSTANCES, OVERSIZED_DEMAND


class InfeasibleInstanceError(Exception):
    """Raised when an instance cannot have any feasible solution."""


@dataclass
class Instance:
    """One CFLP benchmark instance. Facility index i, customer index j."""
    name: str
    capacity: np.ndarray     # S_i, shape (m,)
    fixed_cost: np.ndarray   # F_i, shape (m,)   -> used in objective f1
    demand: np.ndarray       # d_j, shape (n,)
    cost: np.ndarray         # C_ij, shape (m, n) -> used in objective f2
    # original_customer[j] = customer number in the cap file (1-based).
    # Equal to j + 1 unless oversized customers were split.
    original_customer: np.ndarray

    @property
    def m(self):
        return len(self.capacity)

    @property
    def n(self):
        return len(self.demand)


# -----------------------------------------------------------------------------
# 1. Data integrity
# -----------------------------------------------------------------------------
def verify_data_integrity():
    """Check every file listed in SHA256SUMS against its stored hash.

    A mismatch usually means Git converted line endings (LF -> CRLF) on
    checkout, or a file was edited. Raises ValueError listing every mismatch.
    """
    sums_path = DATA_DIR / CHECKSUM_FILE
    if not sums_path.exists():
        raise FileNotFoundError(f"Checksum file not found: {sums_path}")

    mismatches = []
    for line in sums_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, file_name = line.split(maxsplit=1)
        file_name = file_name.lstrip("*")  # sha256sum marks binary mode with '*'
        file_path = DATA_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File listed in {CHECKSUM_FILE} is missing: {file_path}")

        content = file_path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual.lower() != expected.lower():
            hint = " (contains CRLF line endings: Git likely converted them)" if b"\r\n" in content else ""
            mismatches.append(f"  {file_name}: expected {expected[:12]}..., got {actual[:12]}...{hint}")

    if mismatches:
        raise ValueError("Data files do not match SHA256SUMS:\n" + "\n".join(mismatches))
    print(f"All data files in {DATA_DIR} match {CHECKSUM_FILE}.")


# -----------------------------------------------------------------------------
# 2. Parsing
# -----------------------------------------------------------------------------
def load_instance(name):
    """Read data/orlib/<name>.txt and return an Instance. Values are not changed."""
    path = DATA_DIR / f"{name}.txt"
    numbers = path.read_text().split()  # all values in order, line breaks ignored

    m, n = int(numbers[0]), int(numbers[1])
    expected_count = 2 + 2 * m + n * (1 + m)
    if len(numbers) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} values for m={m}, n={n}, "
                         f"found {len(numbers)}")

    values = np.array(numbers[2:], dtype=float)  # float() also reads '7500.' and '.00000'

    facility_part = values[:2 * m].reshape(m, 2)        # one row per facility
    customer_part = values[2 * m:].reshape(n, 1 + m)    # one row per customer

    return Instance(
        name=name,
        capacity=facility_part[:, 0],
        fixed_cost=facility_part[:, 1],
        demand=customer_part[:, 0],
        cost=customer_part[:, 1:].T,      # file stores costs per customer; transpose to C[i, j]
        original_customer=np.arange(1, n + 1),
    )


# -----------------------------------------------------------------------------
# 3. Feasibility check and handling of oversized customers
# -----------------------------------------------------------------------------
def prepare_instance(inst):
    """Make sure a feasible single-sourcing solution can exist.

    Returns the instance unchanged when every customer fits in some facility.
    Otherwise follows OVERSIZED_DEMAND from config.yaml:
      report_infeasible -> raise InfeasibleInstanceError with the proof
      split             -> split each oversized customer into capacity-sized parts
    """
    total_demand, total_capacity = inst.demand.sum(), inst.capacity.sum()
    if total_demand > total_capacity:
        raise InfeasibleInstanceError(
            f"{inst.name}: total demand {total_demand:,.0f} exceeds total capacity {total_capacity:,.0f}.")

    largest_capacity = inst.capacity.max()
    oversized = np.flatnonzero(inst.demand > largest_capacity)
    if len(oversized) == 0:
        return inst

    details = ", ".join(f"customer {j + 1} (demand {inst.demand[j]:,.0f})" for j in oversized)
    if OVERSIZED_DEMAND == "report_infeasible":
        raise InfeasibleInstanceError(
            f"{inst.name} has no feasible solution: {details} exceed(s) the largest facility "
            f"capacity of {largest_capacity:,.0f}, but every customer must be served by exactly one facility.")
    if OVERSIZED_DEMAND == "split":
        print(f"WARNING: {inst.name}: splitting {details} into parts of at most {largest_capacity:,.0f}.")
        return split_oversized_customers(inst, largest_capacity)
    raise ValueError(f"Unknown oversized_demand setting in config.yaml: {OVERSIZED_DEMAND!r}")


def split_oversized_customers(inst, part_size):
    """Replace every customer with demand > part_size by several smaller customers.

    Example: demand 12,912 with part_size 5,000 -> parts 5,000 + 5,000 + 2,912.
    Each part's cost is the original cost times its share of the demand,
    because C_ij is the cost of serving ALL of customer j's demand.
    """
    demands, cost_columns, origins = [], [], []
    for j in range(inst.n):
        d = inst.demand[j]
        n_parts = math.ceil(d / part_size)
        for k in range(n_parts):
            part = min(part_size, d - k * part_size)
            demands.append(part)
            cost_columns.append(inst.cost[:, j] * part / d)
            origins.append(inst.original_customer[j])

    return Instance(
        name=inst.name,
        capacity=inst.capacity,
        fixed_cost=inst.fixed_cost,
        demand=np.array(demands),
        cost=np.column_stack(cost_columns),
        original_customer=np.array(origins),
    )


def get_instance(name):
    """Load an instance and prepare it. This is what other modules should call."""
    return prepare_instance(load_instance(name))


# -----------------------------------------------------------------------------
# Self-check: python -m src.data_loader
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    verify_data_integrity()
    for category, name in INSTANCES.items():
        inst = load_instance(name)
        print(f"\n{category}: {name}  ({inst.m} facilities, {inst.n} customers)")
        print(f"  capacity     min {inst.capacity.min():,.0f}  max {inst.capacity.max():,.0f}")
        print(f"  fixed cost   min {inst.fixed_cost.min():,.0f}  max {inst.fixed_cost.max():,.0f}")
        print(f"  demand       total {inst.demand.sum():,.0f}  max {inst.demand.max():,.0f}")
        try:
            prepared = prepare_instance(inst)
            print(f"  OK: ready for the MOEAs ({prepared.n} customers after preparation)")
        except InfeasibleInstanceError as error:
            print(f"  INFEASIBLE: {error}")