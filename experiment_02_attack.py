"""Experiment B: ordinary consensus with a fixed-value attacker."""
# Consensus background: Olfati-Saber, Fax & Murray (2007).
# https://www.cds.caltech.edu/~murray/papers/2007c_ofm07-ieeeproc.html
import numpy as np
from experiment_common import experiment_cli


def update_node(own, neighbor_values, blend=0.25):
    """Use the same update as A, without knowing the attacker's identity."""
    inclusive_values = np.column_stack((own, neighbor_values))
    return (1.0 - blend) * own + blend * inclusive_values.mean(axis=1)


if __name__ == "__main__":
    experiment_cli("02_attack", update_node, attack=True)
