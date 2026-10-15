# Local Simulation Results

Generated from saved official Robotarium simulations. Hardware tests are pending.

The cases share initial estimates, poses, graph, seed, controller, and duration.
Primary metrics use nodes 1-7. Node 8 is normal in A and becomes malicious after 5 seconds in B/C.

Robots approach their negotiated targets from dispersed starts. The safety filter maintains separation.

| Experiment | Initial group radius (m) | Final radius (m) | Radius reduction | Max target violation (m) | Min collision-center distance (m) | Diagnostics |
|---|---:|---:|---:|---:|---:|---|
| A  Normal consensus | 1.003 | 0.384 | 61.8% | 0.00000 | 0.24503 | none |
| B  Consensus + attack | 1.003 | 0.330 | 67.1% | 0.31099 | 0.24635 | none |
| C  W-MSR + attack | 1.003 | 0.289 | 71.2% | 0.00000 | 0.24477 | none |

Final target displacement from A: B = 1.32077 m; C = 0.03480 m.
The C-A difference also reflects filtering weights, so it is not solely an attack effect.

Final physical group-center displacement from A: B = 1.065 m; C = 0.080 m.

## Interpretation

- A small reference diameter means agreement; validity also requires staying within the initial normal-node range.
- Validity uses the coordinate-wise initial reference box, separately from arena safety and general convex-hull validity.
- Group radius is the largest body-center distance from the group centroid; contraction measures physical gathering.
- Tracking error uses the control point 0.03 m ahead of each robot. Safe separation leaves a nonzero final error.
- Gathering estimates start from dispersed control points and evolve separately from measured body poses.
- Collision centers are shifted 0.025 m along each heading; the simulator threshold is 0.135 m.
- Safety-filter activity and warnings are recorded in each summary.json.
- Results cover K8, one attacker, synchronous communication, and ideal pose feedback.
- Official initialization remains enabled; trajectory metrics cover the experiment phase.

## Files

- comparison.png: four-metric comparison.
- comparison.gif: accelerated replay of saved simulator poses.
- Per-case overview.png: reference estimates and body trajectories.
- Per-case trajectory.npz / states.csv: raw states.
- Per-case metrics.csv / summary.json: metrics and settings.
