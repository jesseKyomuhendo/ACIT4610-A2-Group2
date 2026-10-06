"""Loads the OR-Library cap files, checks they are unmodified and prepares them for the MOEAs.

Other modules only need get_instance(name).
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
    """One CFLP benchmark instance with facility index i and customer index j."""
    name: str
    capacity: np.ndarray     # S_i with one value per facility
    fixed_cost: np.ndarray   # F_i with one value per facility, used in objective f1
    demand: np.ndarray       # d_j with one value per customer
    cost: np.ndarray         # C_ij as an m by n matrix, used in objective f2
    # Customer number in the cap file, 1-based
    # It equals j + 1 unless oversized customers were split
    original_customer: np.ndarray

    @property
    def m(self):
        """Number of facilities."""
        return len(self.capacity)

    @property
    def n(self):
        """Number of customers."""
        return len(self.demand)


# 1. Data integrity
def verify_data_integrity():
    """Checks every file listed in SHA256SUMS against its stored hash.

    A mismatch usually means Git changed the line endings or a file was edited.
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
        file_name = file_name.lstrip("*")  # sha256sum marks binary mode with a star
        file_path = DATA_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File listed in {CHECKSUM_FILE} is missing: {file_path}")

        content = file_path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual.lower() != expected.lower():
            # Windows line endings in the file are the most common cause
            hint = " (contains CRLF line endings: Git likely converted them)" if b"\r\n" in content else ""
            mismatches.append(f"  {file_name}: expected {expected[:12]}..., got {actual[:12]}...{hint}")

    if mismatches:
        raise ValueError("Data files do not match SHA256SUMS:\n" + "\n".join(mismatches))
    print(f"All data files in {DATA_DIR} match {CHECKSUM_FILE}.")


# 2. Parsing
def load_instance(name):
    """Reads one cap file into an Instance without changing any value."""
    path = DATA_DIR / f"{name}.txt"
    numbers = path.read_text().split()  # all values in order, line breaks have no meaning

    # The file holds m and n, then capacity and fixed cost per facility,
    # then demand and m allocation costs per customer
    m, n = int(numbers[0]), int(numbers[1])
    expected_count = 2 + 2 * m + n * (1 + m)
    if len(numbers) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} values for m={m}, n={n}, "
                         f"found {len(numbers)}")

    values = np.array(numbers[2:], dtype=float)  # float also reads values like 7500. and .00000

    facility_part = values[:2 * m].reshape(m, 2)        # one row per facility
    customer_part = values[2 * m:].reshape(n, 1 + m)    # one row per customer

    return Instance(
        name=name,
        capacity=facility_part[:, 0],
        fixed_cost=facility_part[:, 1],
        demand=customer_part[:, 0],
        cost=customer_part[:, 1:].T,      # the file lists costs per customer, so transpose to C[i, j]
        original_customer=np.arange(1, n + 1),
    )


# 3. Feasibility check and handling of oversized customers
def prepare_instance(inst):
    """Makes sure a feasible solution with one facility per customer can exist.

    Oversized customers are handled as set by OVERSIZED_DEMAND in config.yaml.
    """
    total_demand, total_capacity = inst.demand.sum(), inst.capacity.sum()
    if total_demand > total_capacity:
        raise InfeasibleInstanceError(
            f"{inst.name}: total demand {total_demand:,.0f} exceeds total capacity {total_capacity:,.0f}.")

    # A customer larger than every facility can never be served by one facility
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
    """Replaces every customer larger than part_size by several smaller customers.

    For example demand 12,912 with part_size 5,000 becomes 5,000 and 5,000 and 2,912.
    """
    demands, cost_columns, origins = [], [], []
    for j in range(inst.n):
        d = inst.demand[j]
        n_parts = math.ceil(d / part_size)
        for k in range(n_parts):
            part = min(part_size, d - k * part_size)
            demands.append(part)
            # C_ij covers all of the demand, so each part pays its share of it
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
    """Loads and prepares an instance. This is what other modules should call."""
    return prepare_instance(load_instance(name))


# Self-check that verifies the data and summarises every instance
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