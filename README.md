# Resilient Multi-Robot Gathering

Compare ordinary consensus and W-MSR under a single malicious node using Georgia Tech's Robotarium Python simulator. Eight robots start at dispersed positions, negotiate a gathering point, and approach it while maintaining safe separation.

Local simulations are complete. Hardware testing is in progress; results from a diagnostic rerun are pending.

## Experiments

| Script | Scenario |
|---|---|
| `experiment_01_normal.py` | Eight normal nodes gather near a common target. |
| `experiment_02_attack.py` | Node 8 broadcasts a fixed false target after 5 seconds, pulling ordinary consensus away. |
| `experiment_03_wmsr.py` | W-MSR filters the same attack; the seven normal robots gather near the center. |

All experiments share initial poses and motion constraints. Each runs for about 50 simulated seconds after initialization.

## Run

Requires Python 3.10+ and Git. From the project directory, create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The Robotarium dependency is pinned to the simulator commit used for the local results. No local `vendor/` directory is needed.

Run all experiments and generate results:

```bash
python run_all.py
```

Watch individual experiments, one command at a time:

```bash
python experiment_01_normal.py --show
python experiment_02_attack.py --show
python experiment_03_wmsr.py --show
```

`INITIALIZING` is the drive-in phase. The experiment starts when initialization finishes.

## Results

Results are saved in `outputs/`:

- `comparison.gif`: side-by-side motion replay.
- `comparison.png`: gathering, group location, and target validity.
- `RESULTS.md`: results and interpretation.
- Experiment subfolders: trajectory plots, raw states, metrics, and settings.

## Method

Ordinary consensus blends each node's estimate with the mean of its own and its neighbors' estimates. W-MSR first removes up to one largest neighbor value above self and one smallest below self, separately for each coordinate.

A central process simulates eight nodes on the complete graph K8. Gathering estimates are updated separately from measured poses; each robot tracks its estimate. The safety filter keeps physical robots apart, so they gather around the target rather than overlap at one point.

## Robotarium submission

The three experiment scripts in `submissions/` are standalone. Create one experiment per script on the Robotarium website and follow its validation and submission steps.

After changing the code, rerun the experiments, then rebuild and check the submission scripts:

```bash
python build_submissions.py
python validate_project.py
python verify_submissions.py
```

## Acknowledgments

Built on the [Robotarium Python Simulator](https://github.com/robotarium/robotarium_python_simulator)
for simulation, kinematics, and collision avoidance. The upstream MIT license is
preserved in `THIRD_PARTY_NOTICES.md` and the simulator's original `LICENSE`.
Consensus and W-MSR follow published methods; see `REFERENCES.md`.
