"""Performance metrics: normalisation, hypervolume and front size.

Normalisation procedure (identical for both MOEAs)
--------------------------------------------------
For each instance, the ideal and nadir points are taken over ALL final fronts
that are compared (both algorithms, all configurations, all runs):
    ideal = [min f1, min f2]     nadir = [max f1, max f2]
Every point is then scaled to [0, 1] per objective:
    f_scaled = (f - ideal) / (nadir - ideal)
Because both algorithms use the same ideal and nadir, their hypervolumes are
directly comparable.

Hypervolume (larger is better)
------------------------------
The area of objective space dominated by a front and bounded by the reference
point (HV_REFERENCE_POINT in config.yaml, e.g. [1.1, 1.1] in scaled space,
which is worse than every scaled point). For two objectives it is computed
exactly by sorting the front by f1 and adding up rectangles.
"""

import numpy as np

from src.objectives import non_dominated


def normalization_bounds(fronts):
    """Return (ideal, nadir) over a list of fronts, each an array of shape (k, 2)."""
    all_points = np.vstack(fronts)
    return all_points.min(axis=0), all_points.max(axis=0)


def normalize(points, ideal, nadir):
    """Scale points so that ideal -> 0 and nadir -> 1 in every objective."""
    span = np.where(nadir > ideal, nadir - ideal, 1.0)   # avoid dividing by zero
    return (np.asarray(points) - ideal) / span


def hypervolume_2d(points, reference_point):
    """Exact hypervolume of a set of 2-objective points (minimisation).

    1. Keep the non-dominated points that are better than the reference point.
    2. Sort them by f1 (ascending); f2 is then descending.
    3. Each point adds a rectangle reaching right to the next point's f1
       (or the reference point for the last one) and up to the reference f2.

        f2
        ^   ref ----------------+
        |   |###|               |
        |   *###|#####          |
        |       *#####|######   |
        |             *#########|
        |                 *#####|
        +-------------------------> f1
    """
    ref = np.asarray(reference_point, dtype=float)
    points = np.asarray(points, dtype=float)
    points = points[np.all(points < ref, axis=1)]
    if len(points) == 0:
        return 0.0

    front = non_dominated(points)                 # unique, sorted by f1
    next_f1 = np.append(front[1:, 0], ref[0])     # right edge of each rectangle
    widths = next_f1 - front[:, 0]
    heights = ref[1] - front[:, 1]
    return float(np.sum(widths * heights))


def count_non_dominated(front):
    """Number of distinct non-dominated objective vectors in a final front."""
    return len(non_dominated(front))