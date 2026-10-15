"""Shared setup, control, and logging for collision-safe gathering."""
# Robotarium supplies simulation, mappings, and safety filters (MIT).
# Sources: REFERENCES.md; license: THIRD_PARTY_NOTICES.md.
from pathlib import Path
import argparse
import json
import os
import random
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parent
N = 8
DT = 0.033
DURATION = 50.0
ATTACK_START = 5.0
ATTACKER = 7                 # Python index 7 = robot 8; scenario only.
ATTACK_TARGET = np.array([1.30, 0.65])
PROJECTION = 0.03
SAFETY_RADIUS = 0.17
MAX_SI_SPEED = 0.08
BLEND = 0.25
COMMUNICATION_EVERY = 3      # 0.099 simulated seconds per synchronous round.


def initial_state(seed=7):
    """Create the same dispersed start for all three experiments."""
    rng = np.random.default_rng(seed)
    angles = np.array([-2.7, -1.8, -0.9, 0., 0.9, 1.8, 2.7])
    q = np.zeros((2, N))
    q[:, :7] = np.vstack((np.cos(angles), 0.60 * np.sin(angles)))
    q[:, :7] += rng.uniform(-0.02, 0.02, (2, 7))
    # Node 8 starts in the upper right.
    q[:, ATTACKER] = np.array([0.95, 0.50]) + rng.uniform(-0.01, 0.01, 2)
    headings = np.arctan2(-q[1], -q[0])
    poses = np.vstack((q, headings))
    # Place each forward control point at q_i.
    poses[:2] -= PROJECTION * np.vstack((np.cos(headings), np.sin(headings)))
    return q, poses


def complete_neighbors():
    """Return the K8 neighbor lists without self-edges."""
    return [np.array([j for j in range(N) if j != i]) for i in range(N)]


def consensus_round(q, update_node, neighbors, active_attack, attack_value=None):
    """Update all nodes from the same snapshot."""
    messages = q.copy()
    if active_attack:
        messages[:, ATTACKER] = ATTACK_TARGET if attack_value is None else attack_value
    next_q = messages.copy()
    for i in range(N):
        if active_attack and i == ATTACKER:
            continue
        next_q[:, i] = update_node(messages[:, i], messages[:, neighbors[i]], BLEND)
    return next_q


def create_motion_functions():
    """Create the motion mapping and safety filter."""
    from rps.utilities.transformations import create_si_to_uni_mapping
    from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
    si_to_uni, uni_to_si = create_si_to_uni_mapping(projection_distance=PROJECTION)
    barrier = create_uni_barrier_certificate_with_boundary(
        safety_radius=SAFETY_RADIUS,
        projection_distance=PROJECTION,
        velocity_magnitude_limit=MAX_SI_SPEED,
        boundary_points=np.array([-1.5, 1.5, -0.9, 0.9]),
    )
    return si_to_uni, uni_to_si, barrier


def motion_command(q, x, si_to_uni, uni_to_si, barrier, base_length=0.11):
    """Track gathering estimates and enforce motion limits."""
    targets = q
    desired_si = 0.8 * (targets - uni_to_si(x))
    norms = np.linalg.norm(desired_si, axis=0)
    desired_si *= np.minimum(1.0, MAX_SI_SPEED / np.maximum(norms, 1e-12))
    requested = si_to_uni(desired_si, x)
    command = barrier(requested, x)
    if not np.isfinite(command).all():
        raise RuntimeError("Non-finite safety-filter command")
    # Uniform scaling preserves the CBF inequalities inside the safe set.
    wheel_equivalent = np.max(np.abs(command[0]) + base_length / 2 * np.abs(command[1]))
    command *= min(1.0, 0.18 / max(float(wheel_equivalent), 1e-12))
    return desired_si, command, float(np.linalg.norm(command - requested))


def configure_plotting(show=False):
    # Keep the font cache local.
    cache = ROOT / "work" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    if not show:
        matplotlib.use("Agg")


def point_min_distance(points):
    diff = points[:, :, None] - points[:, None, :]
    distances = np.linalg.norm(diff, axis=0)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def metrics_from_data(data):
    """Compare nodes 1-7 in every experiment."""
    q = data["references"][:, :, :7]
    initial = data["initial_references"][:, :7]
    low, high = initial.min(axis=1), initial.max(axis=1)
    violation = np.maximum(low[None, :, None] - q, q - high[None, :, None])
    outside = np.maximum(violation, 0).max(axis=(1, 2))
    diffs = q[:, :, :, None] - q[:, :, None, :]
    diameter = np.linalg.norm(diffs, axis=1).max(axis=(1, 2))
    poses = data["poses"]
    projected = poses[:, :2] + PROJECTION * np.stack(
        (np.cos(poses[:, 2]), np.sin(poses[:, 2])), axis=1
    )
    tracking = np.linalg.norm(
        projected[:, :, :7] - q, axis=1
    ).mean(axis=1)
    centroid_drift = np.linalg.norm(q.mean(axis=2) - initial.mean(axis=1), axis=1)
    body = poses[:, :2, :7]
    body_centroid = body.mean(axis=2)
    cluster_radius = np.linalg.norm(body - body_centroid[:, :, None], axis=1).max(axis=1)
    body_diameter = np.linalg.norm(body[:, :, :, None] - body[:, :, None, :], axis=1).max(axis=(1, 2))
    body_centroid_drift = np.linalg.norm(body_centroid - body_centroid[0], axis=1)
    return {
        "diameter": diameter,
        "box_violation": outside,
        "tracking_error": tracking,
        "centroid_drift": centroid_drift,
        "cluster_radius": cluster_radius,
        "body_diameter": body_diameter,
        "body_centroid_drift": body_centroid_drift,
    }


def run_experiment(name, update_node, *, attack, seed=7, seconds=DURATION,
                   show=False, fast=False, skip_initialization=False,
                   output_root=None):
    """Run and record an experiment in the official simulator."""
    configure_plotting(show)
    import rps.robotarium as robotarium

    if not np.isfinite(seconds) or seconds <= ATTACK_START + 1:
        raise ValueError("--seconds must be finite and greater than 6 seconds.")
    output_root = Path(output_root) if output_root else ROOT / "outputs"
    output = output_root / name
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)         # Official initialization uses Python random too.
    np.random.seed(seed)
    q, initial_poses = initial_state(seed)
    q_initial = q.copy()
    neighbors = complete_neighbors()
    print(f"[{name}] Creating 8 robots (seed={seed}, initialization={not skip_initialization})", flush=True)
    wall_start = time.perf_counter()
    r = robotarium.Robotarium(
        number_of_robots=N,
        show_figure=show,
        initial_conditions=initial_poses,
        sim_in_real_time=bool(show and not fast),
        skip_initialization=skip_initialization,
    )
    initialization_wall_seconds = time.perf_counter() - wall_start
    si_to_uni, uni_to_si, barrier = create_motion_functions()
    steps = int(np.ceil(seconds / DT))
    times = np.arange(steps + 1) * DT
    poses_log = np.zeros((steps + 1, 3, N))
    references = np.zeros((steps + 1, 2, N))
    desired_log = np.zeros((steps, 2, N))
    command_log = np.zeros((steps, 2, N))
    safety_change = np.zeros(steps)
    qp_warnings = []
    print(f"[{name}] Running {steps} control steps ({steps * DT:.2f} simulated seconds)", flush=True)
    for k in range(steps):
        x = r.get_poses()
        # Log the state before applying this step's command.
        poses_log[k] = x
        references[k] = q
        if k % COMMUNICATION_EVERY == 0:
            q = consensus_round(q, update_node, neighbors, attack and times[k] >= ATTACK_START)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                desired_si, command, change = motion_command(q, x, si_to_uni, uni_to_si, barrier, r.BASE_LENGTH)
            except RuntimeError:
                r.set_velocities(np.arange(N), np.zeros((2, N)))
                r.step()
                raise
        for item in captured:
            qp_warnings.append({"step": k, "message": str(item.message)})
        safety_change[k] = change
        desired_log[k] = desired_si
        command_log[k] = command
        r.set_velocities(np.arange(N), command)
        r.step()
        if k and k % 300 == 0:
            print(f"[{name}] t={times[k]:.1f}s", flush=True)
    # Record the final state and stop.
    x = r.get_poses()
    poses_log[-1] = x
    references[-1] = q
    r.set_velocities(np.arange(N), np.zeros((2, N)))
    r.step()
    r.debug()
    data = {
        "time": times,
        "poses": poses_log,
        "references": references,
        "initial_references": q_initial,
        "initial_poses": initial_poses,
        "desired_si": desired_log,
        "commands": command_log,
        "safety_change": safety_change,
    }
    np.savez_compressed(output / "trajectory.npz", **data)
    series = metrics_from_data(data)
    centers = poses_log[:, :2] + r.COLLISION_OFFSET * np.stack(
        (np.cos(poses_log[:, 2]), np.sin(poses_log[:, 2])), axis=1
    )
    min_collision = min(point_min_distance(p) for p in centers)
    min_axle = min(point_min_distance(p) for p in poses_log[:, :2])
    bounds = r.BOUNDARIES
    xy = poses_log[:, :2]
    clearance = min(
        float((xy[:, 0] - bounds[0]).min()), float((bounds[1] - xy[:, 0]).min()),
        float((xy[:, 1] - bounds[2]).min()), float((bounds[3] - xy[:, 1]).min()),
    )
    wheel_max = np.max(np.abs(command_log[:, 0]) + r.BASE_LENGTH / 2 * np.abs(command_log[:, 1]))
    summary = {
        "experiment": name, "backend": "official Robotarium Python simulator (unicycle)",
        "seed": seed, "robots": N, "graph": "complete K8", "attack": attack,
        "attack_robot_number": 8 if attack else None,
        "attack_start_seconds": ATTACK_START if attack else None,
        "attack_target": ATTACK_TARGET.tolist() if attack else None,
        "duration_seconds": float(times[-1]), "communication_period_seconds": DT * COMMUNICATION_EVERY,
        "blend": BLEND, "skip_initialization": skip_initialization,
        "initialization_wall_seconds": initialization_wall_seconds,
        "final_reference_mean_nodes_1_to_7": references[-1, :, :7].mean(axis=1).tolist(),
        "final_diameter_m": float(series["diameter"][-1]),
        "max_box_violation_m": float(series["box_violation"].max()),
        "final_centroid_drift_m": float(series["centroid_drift"][-1]),
        "final_mean_tracking_error_m": float(series["tracking_error"][-1]),
        "initial_cluster_radius_m": float(series["cluster_radius"][0]),
        "final_cluster_radius_m": float(series["cluster_radius"][-1]),
        "cluster_radius_reduction_fraction": float(1 - series["cluster_radius"][-1] / series["cluster_radius"][0]),
        "final_body_diameter_m": float(series["body_diameter"][-1]),
        "final_body_centroid_nodes_1_to_7": poses_log[-1, :2, :7].mean(axis=1).tolist(),
        "final_body_centroid_drift_m": float(series["body_centroid_drift"][-1]),
        "minimum_axle_distance_m": min_axle,
        "minimum_collision_center_distance_m": min_collision,
        "simulator_collision_threshold_m": float(r.COLLISION_DIAMETER),
        "minimum_axle_boundary_clearance_m": clearance,
        "max_wheel_equivalent_speed_m_s": float(wheel_max),
        "safety_filter_modified_steps": int(np.count_nonzero(safety_change > 1e-4)),
        "safety_filter_warning_count": len(qp_warnings), "safety_filter_warnings": qp_warnings,
        # Store the simulator's diagnostics alongside measured metrics.
        "official_diagnostics": dict(r._errors),
        "wall_seconds": time.perf_counter() - wall_start,
        "hardware_tested": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv = np.column_stack((times, *(series[key] for key in (
        "diameter", "box_violation", "tracking_error", "centroid_drift",
        "cluster_radius", "body_diameter", "body_centroid_drift"
    ))))
    np.savetxt(output / "metrics.csv", csv, delimiter=",", comments="",
               header="time_s,reference_diameter_m,box_violation_m,mean_tracking_error_m,centroid_drift_m,cluster_radius_m,body_diameter_m,body_centroid_drift_m")
    rows = []
    for k, t in enumerate(times):
        for i in range(N):
            rows.append([t, i + 1, *poses_log[k, :, i], *references[k, :, i]])
    np.savetxt(output / "states.csv", rows, delimiter=",", comments="",
               header="time_s,robot_number,x_m,y_m,heading_rad,reference_x_m,reference_y_m")
    print(f"[{name}] Saved {output}", flush=True)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return summary


def experiment_cli(name, update_node, attack):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Open live Robotarium animation")
    parser.add_argument("--fast", action="store_true", help="Do not throttle a visible simulation")
    parser.add_argument("--seconds", type=float, default=DURATION)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-initialization", action="store_true",
                        help="DEBUG ONLY: bypass official drive-in; not submission verification")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    run_experiment(name, update_node, attack=attack, seed=args.seed, seconds=args.seconds,
                   show=args.show, fast=args.fast, skip_initialization=args.skip_initialization,
                   output_root=args.output)
