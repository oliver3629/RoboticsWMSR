"""Experiment C: coordinate-wise W-MSR with the same attacker as B."""
# Method: Zhang & Sundaram, arXiv:1110.3843v2 (2012), Section III-B.
# https://arxiv.org/abs/1110.3843v2
# Adaptation here: separate x/y updates, relaxation blend=0.25, F=1.
import numpy as np
from experiment_common import experiment_cli


def update_node(own, neighbor_values, blend=0.25, f=1):
    """Trim up to F extremes on each side of self, then blend the mean."""
    result = own.copy()
    for coordinate in range(len(own)):
        values = neighbor_values[coordinate]
        lower = np.flatnonzero(values < own[coordinate])
        upper = np.flatnonzero(values > own[coordinate])
        keep = np.ones(values.size, dtype=bool)
        if f > 0:
            lower_order = lower[np.argsort(values[lower], kind="stable")]
            upper_order = upper[np.argsort(values[upper], kind="stable")]
            keep[lower_order[:f]] = False
            keep[upper_order[-f:]] = False
        retained = np.r_[own[coordinate], values[keep]]
        result[coordinate] = (
            (1.0 - blend) * own[coordinate] + blend * retained.mean()
        )
    return result


if __name__ == "__main__":
    experiment_cli("03_wmsr", update_node, attack=True)
