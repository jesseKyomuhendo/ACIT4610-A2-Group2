"""Objective evaluation (computed only after a feasible assignment is produced).

f1 = sum_i F_i y_i          (facility-opening cost)
f2 = sum_i sum_j C_ij x_ij  (customer-allocation cost; C_ij NOT multiplied by d_j)
Also count objective evaluations here so both MOEAs share the same budget.
"""
# TODO: implement
