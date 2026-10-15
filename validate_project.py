"""Check algorithm invariants, safety metrics, and submission consistency."""
import ast
import itertools
import json
from pathlib import Path

import numpy as np

from experiment_01_normal import update_node as ordinary
from experiment_03_wmsr import update_node as wmsr
from experiment_common import ROOT, N, initial_state, complete_neighbors, consensus_round
from analyze_results import CASES, load_results


def check_filter_edges():
    # Self is the minimum: no lower value should be removed; reject only upper max.
    assert np.allclose(wmsr(np.array([0.]), np.array([[1., 2., 10.]]), blend=1), [1.])
    # Self is the maximum: no upper value should be removed.
    assert np.allclose(wmsr(np.array([10.]), np.array([[-10., 1., 2.]]), blend=1), [13 / 3])
    # Ties equal to self stay. F=0 reduces to the ordinary update.
    assert np.allclose(wmsr(np.array([2.]), np.array([[2., 2., 100.]]), blend=1), [2.])
    own, values = np.array([.1, -.2]), np.array([[2., .3, -.1], [-1., .8, .1]])
    assert np.allclose(wmsr(own, values, f=0), ordinary(own, values))


def check_graph():
    """Exhaustively check 3-robustness: every disjoint nonempty pair of subsets."""
    neighbors = complete_neighbors()
    checked = 0
    for labels in itertools.product(range(3), repeat=N):
        first = {i for i, label in enumerate(labels) if label == 1}
        second = {i for i, label in enumerate(labels) if label == 2}
        if not first or not second:
            continue
        reachable_a = any(len(set(neighbors[i]) - first) >= 3 for i in first)
        reachable_b = any(len(set(neighbors[i]) - second) >= 3 for i in second)
        assert reachable_a or reachable_b
        checked += 1
    return checked


def check_adversaries():
    """Logical-layer stress tests, explicitly not extra physical robot simulations."""
    neighbors = complete_neighbors()
    rng = np.random.default_rng(914)
    checked = 0
    largest_diameter = 0.
    for seed in range(30):
        for attack in ("constant", "alternating", "inside_current_range", "random_large"):
            q, _ = initial_state(seed)
            low, high = q[:, :7].min(axis=1), q[:, :7].max(axis=1)
            for step in range(220):
                if attack == "constant":
                    value = np.array([10., -10.])
                elif attack == "alternating":
                    value = (-1) ** step * np.array([10., 10.])
                elif attack == "inside_current_range":
                    value = .99 * q[:, :7].max(axis=1) + .01 * q[:, :7].min(axis=1)
                else:
                    value = rng.uniform(-100, 100, 2)
                q = consensus_round(q, wmsr, neighbors, True, value)
                assert np.isfinite(q).all()
                assert np.all(q[:, :7] >= low[:, None] - 1e-12)
                assert np.all(q[:, :7] <= high[:, None] + 1e-12)
            diameter = np.max(np.linalg.norm(q[:, :7, None] - q[:, None, :7], axis=0))
            assert diameter < 1e-6
            largest_diameter = max(largest_diameter, float(diameter))
            checked += 1
    return checked, largest_diameter


def check_submission_sources():
    """The generated controls must be the same AST as the locally tested controls."""
    common_tree = ast.parse((ROOT / "experiment_common.py").read_text())
    names = {"initial_state", "complete_neighbors", "consensus_round", "create_motion_functions", "motion_command"}
    expected = {node.name: ast.dump(node, include_attributes=False) for node in common_tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names}
    for case in CASES:
        generated = ROOT / "submissions" / f"experiment_{case}.py"
        tree = ast.parse(generated.read_text())
        functions = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                     if isinstance(node, ast.FunctionDef)}
        for name, value in expected.items():
            assert functions[name] == value
        source = ast.parse((ROOT / f"experiment_{case}.py").read_text())
        update = next(node for node in source.body if isinstance(node, ast.FunctionDef))
        assert functions["update_node"] == ast.dump(update, include_attributes=False)


def check_saved_results():
    data, summaries = load_results(ROOT / "outputs")
    assert all(not s["skip_initialization"] for s in summaries)
    for values, s in zip(data, summaries):
        assert all(np.isfinite(array).all() for array in values.values())
        assert s["official_diagnostics"] == {}, s
        assert s["safety_filter_warning_count"] == 0, s
        assert s["minimum_collision_center_distance_m"] > s["simulator_collision_threshold_m"]
        assert s["minimum_axle_boundary_clearance_m"] > .1
        assert s["max_wheel_equivalent_speed_m_s"] <= .18 + 1e-10
        assert s["final_diameter_m"] < 1e-5
        # Physical robots gather around a common target without overlapping.
        # A nonzero tracking error is expected when the CBF prevents overlap.
        assert s["final_mean_tracking_error_m"] < .45
        assert s["cluster_radius_reduction_fraction"] > .5
        assert s["final_cluster_radius_m"] < .5
    assert summaries[0]["max_box_violation_m"] < 1e-10
    assert summaries[1]["max_box_violation_m"] > .15
    assert summaries[2]["max_box_violation_m"] < 1e-10
    # Conditions A/B have identical histories before the attack.
    before = data[0]["time"] < 5
    assert np.array_equal(data[0]["references"][before], data[1]["references"][before])
    assert np.allclose(data[0]["poses"][before], data[1]["poses"][before], atol=1e-10)
    normal_center = np.array(summaries[0]["final_body_centroid_nodes_1_to_7"])
    bad_center = np.array(summaries[1]["final_body_centroid_nodes_1_to_7"])
    good_center = np.array(summaries[2]["final_body_centroid_nodes_1_to_7"])
    assert np.linalg.norm(bad_center - normal_center) > .6
    assert np.linalg.norm(good_center - normal_center) < .2


if __name__ == "__main__":
    check_filter_edges()
    partitions = check_graph()
    cases, diameter = check_adversaries()
    check_submission_sources()
    check_saved_results()
    result = {
        "passed": True, "graph_3_robust_subset_pairs_checked": partitions,
        "logical_layer_stress_cases": cases,
        "logical_layer_seeds": 30, "logical_attack_types": 4,
        "largest_final_logical_diameter_m": diameter,
        "official_physical_simulation_cases_checked": 3,
        "submission_core_matches_local_core": True,
        "hardware_tested": False,
    }
    path = ROOT / "outputs" / "validation.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
