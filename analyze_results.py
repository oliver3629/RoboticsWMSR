"""Generate plots, animation, and an English report from saved results."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiment_common import ROOT, ATTACK_START, ATTACK_TARGET, metrics_from_data, configure_plotting

CASES = ["01_normal", "02_attack", "03_wmsr"]
LABELS = ["A  Normal consensus", "B  Consensus + attack", "C  W-MSR + attack"]
COLORS = ["#2563eb", "#ea580c", "#059669"]


def load_results(root):
    datasets, summaries = [], []
    for case in CASES:
        with np.load(root / case / "trajectory.npz") as values:
            datasets.append({key: values[key].copy() for key in values.files})
        summaries.append(json.loads((root / case / "summary.json").read_text()))
    for other in datasets[1:]:
        if not (np.array_equal(datasets[0]["initial_poses"], other["initial_poses"])
                and np.array_equal(datasets[0]["initial_references"], other["initial_references"])
                and np.array_equal(datasets[0]["time"], other["time"])):
            raise ValueError("Results must share initial states and duration for a fair comparison")
    if len({summary["seed"] for summary in summaries}) != 1:
        raise ValueError("Cannot compare different seeds")
    return datasets, summaries


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.facecolor": "#f8fafc",
        "axes.facecolor": "white", "savefig.facecolor": "#f8fafc",
        "grid.alpha": 0.18,
    })


def comparison_plot(datasets, root):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for data, label, color in zip(datasets, LABELS, COLORS):
        t = data["time"]
        q = data["references"][:, :, :7]
        values = metrics_from_data(data)
        axes[0, 0].plot(t, values["cluster_radius"], color=color, label=label, lw=2.2)
        axes[0, 1].plot(t, data["poses"][:, 0, :7].mean(axis=1), color=color, lw=2)
        axes[1, 0].plot(t, q[:, 0].mean(axis=1), color=color, lw=2)
        axes[1, 1].plot(t, values["box_violation"], color=color, lw=2)
    low = datasets[0]["initial_references"][0, :7].min()
    high = datasets[0]["initial_references"][0, :7].max()
    axes[1, 0].axhspan(low, high, color="#94a3b8", alpha=.2, label="Initial normal-node x range")
    axes[1, 0].axhline(ATTACK_TARGET[0], color="#64748b", ls=":", label="Attacker's x target")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    axes[0, 0].set(title="Physical gathering: the group contracts", ylabel="Max body distance to group centroid (m)")
    axes[0, 1].set(title="Physical location: is the group pulled away?", ylabel="Mean body x, nodes 1-7 (m)")
    axes[1, 0].set(title="Negotiated gathering target", ylabel="Mean target x, nodes 1-7 (m)")
    axes[1, 1].set(title="Target validity: leaving the initial normal-node box", ylabel="Maximum coordinate violation (m)")
    for ax in axes.flat:
        ax.axvline(ATTACK_START, color="#64748b", ls="--", lw=1)
        ax.set_xlabel("Simulated time (s)")
        ax.grid(True)
    fig.suptitle("Resilient robot gathering | dispersed starts, K8, F = 1", fontsize=16, weight="bold")
    fig.savefig(root / "comparison.png", dpi=180)
    plt.close(fig)


def individual_plots(datasets, root):
    import matplotlib.pyplot as plt
    for case, title, data in zip(CASES, LABELS, datasets):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
        t, q, poses = data["time"], data["references"], data["poses"]
        palette = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#0891b2", "#92400e", "#db2777"]
        for i in range(8):
            color = "#dc2626" if i == 7 else palette[i]
            for coordinate in range(2):
                axes[coordinate].plot(t, q[:, coordinate, i], color=color,
                                      ls="--" if i == 7 else "-", lw=1.5,
                                      label=f"Robot {i + 1}")
            axes[2].plot(poses[:, 0, i], poses[:, 1, i], color=color, alpha=.8, lw=1.5)
            axes[2].scatter(*poses[0, :2, i], color=color, marker="o", s=15, alpha=.4)
            axes[2].scatter(*poses[-1, :2, i], color=color, marker="s", s=32)
            axes[2].annotate(str(i + 1), poses[-1, :2, i], xytext=(5, 4), textcoords="offset points", fontsize=8)
        for dim, ax in enumerate(axes[:2]):
            initial = q[0, dim, :7]
            ax.axhspan(initial.min(), initial.max(), color="#94a3b8", alpha=.2)
            ax.axvline(ATTACK_START, color="#64748b", ls="--", lw=1)
            ax.set(xlabel="Simulated time (s)", ylabel=f"Virtual {'xy'[dim]} reference (m)")
            ax.grid(True)
        axes[0].legend(ncol=2, fontsize=7)
        axes[2].set(xlim=(-1.6, 1.6), ylim=(-1, 1), xlabel="Body x (m)", ylabel="Body y (m)",
                    title="Physical trajectories\ncircle: start | square: end", aspect="equal")
        target = q[-1, :, :7].mean(axis=1)
        axes[2].scatter(*target, marker="*", s=100, c="#111827", zorder=5, label="Mean final target")
        axes[2].legend(fontsize=7, loc="lower left")
        axes[2].grid(True)
        fig.suptitle(title + " | simulation only", weight="bold", fontsize=14)
        fig.savefig(root / case / "overview.png", dpi=150)
        plt.close(fig)


def animation(datasets, root):
    """Replay saved official simulator body poses; no interpolation is fabricated."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
    body_colors = ["#2563eb"] * 7 + ["#dc2626"]
    from matplotlib.patches import Circle
    bodies, targets, trails, captions = [], [], [], []
    for ax, title, data in zip(axes, LABELS, datasets):
        ax.set(xlim=(-1.6, 1.6), ylim=(-1, 1), aspect="equal", title=title,
               xlabel="Body x (m)", ylabel="Body y (m)")
        ax.grid(True)
        first = data["poses"][0, :2]
        patches = [Circle(first[:, i], .055, facecolor=body_colors[i], edgecolor="white", lw=.6, zorder=3)
                   for i in range(8)]
        for patch in patches:
            ax.add_patch(patch)
        bodies.append(patches)
        ax.scatter(first[0], first[1], c="#94a3b8", s=12, alpha=.5, zorder=1)
        targets.append(ax.scatter([], [], marker="*", s=70, c="#111827", zorder=4))
        lines = [ax.plot([], [], color=body_colors[i], lw=1, alpha=.4)[0] for i in range(8)]
        trails.append(lines)
        captions.append(ax.text(.02, .02, "", transform=ax.transAxes, fontsize=9))
    sampled = np.unique(np.linspace(0, len(datasets[0]["time"]) - 1, 151).astype(int))
    frames = np.r_[np.zeros(5, dtype=int), sampled, np.full(15, sampled[-1])]
    title = fig.suptitle("Robot gathering | blue: nodes 1-7 | red: node 8 | star: mean target | accelerated replay", fontsize=11)

    def update(frame):
        for index, data in enumerate(datasets):
            p = data["poses"]
            for i, patch in enumerate(bodies[index]):
                patch.center = p[frame, :2, i]
            targets[index].set_offsets(data["references"][frame, :, :7].mean(axis=1).reshape(1, 2))
            for i, line in enumerate(trails[index]):
                line.set_data(p[:frame + 1, 0, i], p[:frame + 1, 1, i])
            t = data["time"][frame]
            captions[index].set_text(f"t = {t:.1f} s | " + (
                "attack active" if index and t >= ATTACK_START else "no active attack"))
        return [title, *targets, *captions]

    movie = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    movie.save(root / "comparison.gif", writer=PillowWriter(fps=10), dpi=85)
    plt.close(fig)


def report(summaries, root):
    lines = [
        "# Local Simulation Results", "",
        "Generated from saved official Robotarium simulations. Hardware tests are pending.", "",
        "The cases share initial estimates, poses, graph, seed, controller, and duration.",
        "Primary metrics use nodes 1-7. Node 8 is normal in A and becomes malicious after 5 seconds in B/C.", "",
        "Robots approach their negotiated targets from dispersed starts. The safety filter maintains separation.", "",
        "| Experiment | Initial group radius (m) | Final radius (m) | Radius reduction | Max target violation (m) | Min collision-center distance (m) | Diagnostics |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for title, s in zip(LABELS, summaries):
        lines.append(f"| {title} | {s['initial_cluster_radius_m']:.3f} | {s['final_cluster_radius_m']:.3f} | "
                     f"{s['cluster_radius_reduction_fraction']:.1%} | {s['max_box_violation_m']:.5f} | "
                     f"{s['minimum_collision_center_distance_m']:.5f} | "
                     f"{s['official_diagnostics'] or 'none'} |")
    bias = np.linalg.norm(np.array(summaries[1]["final_reference_mean_nodes_1_to_7"]) -
                          summaries[0]["final_reference_mean_nodes_1_to_7"])
    resilient_bias = np.linalg.norm(np.array(summaries[2]["final_reference_mean_nodes_1_to_7"]) -
                                    summaries[0]["final_reference_mean_nodes_1_to_7"])
    body_bias = np.linalg.norm(np.array(summaries[1]["final_body_centroid_nodes_1_to_7"]) -
                              summaries[0]["final_body_centroid_nodes_1_to_7"])
    resilient_body_bias = np.linalg.norm(np.array(summaries[2]["final_body_centroid_nodes_1_to_7"]) -
                                        summaries[0]["final_body_centroid_nodes_1_to_7"])
    lines += ["", f"Final target displacement from A: B = {bias:.5f} m; C = {resilient_bias:.5f} m.",
              "The C-A difference also reflects filtering weights, so it is not solely an attack effect.", "",
              f"Final physical group-center displacement from A: B = {body_bias:.3f} m; C = {resilient_body_bias:.3f} m.", "",
              "## Interpretation", "",
              "- A small reference diameter means agreement; validity also requires staying within the initial normal-node range.",
              "- Validity uses the coordinate-wise initial reference box, separately from arena safety and general convex-hull validity.",
              "- Group radius is the largest body-center distance from the group centroid; contraction measures physical gathering.",
              "- Tracking error uses the control point 0.03 m ahead of each robot. Safe separation leaves a nonzero final error.",
              "- Gathering estimates start from dispersed control points and evolve separately from measured body poses.",
              "- Collision centers are shifted 0.025 m along each heading; the simulator threshold is 0.135 m.",
              "- Safety-filter activity and warnings are recorded in each summary.json.",
              "- Results cover K8, one attacker, synchronous communication, and ideal pose feedback.",
              "- Official initialization remains enabled; trajectory metrics cover the experiment phase.", "",
              "## Files", "", "- comparison.png: four-metric comparison.", "- comparison.gif: accelerated replay of saved simulator poses.",
              "- Per-case overview.png: reference estimates and body trajectories.", "- Per-case trajectory.npz / states.csv: raw states.",
              "- Per-case metrics.csv / summary.json: metrics and settings.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(root=ROOT / "outputs", make_gif=True):
    configure_plotting(False)
    style()
    root = Path(root)
    datasets, summaries = load_results(root)
    comparison_plot(datasets, root)
    individual_plots(datasets, root)
    report(summaries, root)
    if make_gif:
        animation(datasets, root)
    print(f"Figures and report saved to {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-gif", action="store_true")
    options = parser.parse_args()
    analyze(options.output, not options.no_gif)
