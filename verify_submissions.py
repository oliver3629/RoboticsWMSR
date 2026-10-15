"""Run standalone scripts headlessly and compare their saved trajectories."""
import json
import os
import runpy

import numpy as np

from experiment_common import ROOT, configure_plotting
from analyze_results import CASES


def verify():
    configure_plotting(False)
    import rps.robotarium as robotarium
    original = robotarium.Robotarium

    def headless_robotarium(**kwargs):
        kwargs["show_figure"] = False
        kwargs["sim_in_real_time"] = False
        return original(**kwargs)

    previous_cwd = os.getcwd()
    results = []
    try:
        robotarium.Robotarium = headless_robotarium
        for case in CASES:
            destination = ROOT / "work" / "submission_checks" / case
            destination.mkdir(parents=True, exist_ok=True)
            os.chdir(destination)
            print(f"Executing standalone candidate: {case}", flush=True)
            runpy.run_path(str(ROOT / "submissions" / f"experiment_{case}.py"), run_name="__main__")
            candidate = np.load(destination / f"experiment_{case}_data.npz")
            local = np.load(ROOT / "outputs" / case / "trajectory.npz")
            maximum_difference = 0.
            for key in ("time", "poses", "references", "commands", "initial_references", "initial_poses"):
                difference = float(np.max(np.abs(candidate[key] - local[key])))
                assert difference < 1e-10, (case, key, difference)
                maximum_difference = max(maximum_difference, difference)
            candidate.close()
            local.close()
            results.append({"case": case, "passed": True, "maximum_absolute_difference": maximum_difference})
    finally:
        robotarium.Robotarium = original
        os.chdir(previous_cwd)
    (ROOT / "outputs" / "submission_validation.json").write_text(json.dumps({
        "passed": True, "local_headless_execution": True,
        "official_initialization_enabled": True, "platform_verified": False,
        "cases": results,
    }, indent=2))
    print("All standalone candidates match the local trajectories.", flush=True)


if __name__ == "__main__":
    verify()
