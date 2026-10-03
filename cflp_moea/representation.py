"""Chromosome representation and feasibility repair / decoding.

Must be shared by BOTH MOEAs (same representation + repair = fair comparison).
Decoding must always yield a feasible solution:
  - every customer assigned to exactly one facility
  - only to opened facilities (x_ij <= y_i)
  - sum_j d_j x_ij <= S_i y_i for every facility i
"""
# TODO: implement
