"""Run all experiments and generate plots, animation, and a report."""
import argparse
from pathlib import Path

from experiment_01_normal import update_node as ordinary
from experiment_02_attack import update_node as attacked
from experiment_03_wmsr import update_node as wmsr
from experiment_common import ROOT, DURATION, run_experiment
from analyze_results import analyze


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=DURATION)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-gif", action="store_true")
    options = parser.parse_args()
    for name, updater, attack in [
        ("01_normal", ordinary, False), ("02_attack", attacked, True), ("03_wmsr", wmsr, True)
    ]:
        run_experiment(name, updater, attack=attack, seed=options.seed,
                       seconds=options.seconds, output_root=options.output)
    analyze(options.output, not options.no_gif)
