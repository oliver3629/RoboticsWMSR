"""Experiment A: ordinary consensus with eight normal nodes."""
# Method background: Olfati-Saber, Fax & Murray (2007), consensus survey.
# https://www.cds.caltech.edu/~murray/papers/2007c_ofm07-ieeeproc.html
import numpy as np
from experiment_common import experiment_cli


def update_node(own, neighbor_values, blend=0.25):
    """Blend the current estimate with the local mean."""
    inclusive_values = np.column_stack((own, neighbor_values))
    return (1.0 - blend) * own + blend * inclusive_values.mean(axis=1)


if __name__ == "__main__":
    experiment_cli("01_normal", update_node, attack=False)
