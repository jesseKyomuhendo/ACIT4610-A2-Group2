"""Performance metrics for the final fronts, namely normalisation, hypervolume and front size.

Both MOEAs use the same normalisation and reference point, so their hypervolumes are comparable.
"""

import numpy as np

from src.objectives import non_dominated


def normalization_bounds(fronts):
    """Returns the ideal and nadir points over a list of fronts.

    Pass all final fronts of an instance, from both algorithms, all configurations and all runs.
    """
    all_points = np.vstack(fronts)
    # Ideal is the best value of each objective and nadir the worst
    return all_points.min(axis=0), all_points.max(axis=0)


def normalize(points, ideal, nadir):
    """Scales points so that ideal becomes 0 and nadir becomes 1 in every objective."""
    span = np.where(nadir > ideal, nadir - ideal, 1.0)   # avoid dividing by zero
    return (np.asarray(points) - ideal) / span


def hypervolume_2d(points, reference_point):
    """Exact hypervolume of two-objective points for minimisation, where larger is better.

    It is the area dominated by the points and bounded by the reference point.
    """
    ref = np.asarray(reference_point, dtype=float)
    points = np.asarray(points, dtype=float)
    # Points beyond the reference point add no area
    points = points[np.all(points < ref, axis=1)]
    if len(points) == 0:
        return 0.0

    # Sorted by f1 ascending, so f2 is descending along the front
    front = non_dominated(points)                 # unique, sorted by f1
    # Each point adds a rectangle reaching right to the next point and up to the reference point
    next_f1 = np.append(front[1:, 0], ref[0])     # right edge of each rectangle
    widths = next_f1 - front[:, 0]
    heights = ref[1] - front[:, 1]
    return float(np.sum(widths * heights))


def count_non_dominated(front):
    """Number of distinct non-dominated objective vectors in a final front."""
    return len(non_dominated(front))