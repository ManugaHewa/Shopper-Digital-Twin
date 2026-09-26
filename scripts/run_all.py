"""Run the whole project from scratch, one stage after another, skipping stages that are already done.

    python scripts/run_all.py                      # everything that isn't done yet
    python scripts/run_all.py --from twin          # redo the twin stage and every stage after it
    python scripts/run_all.py --only twin --force  # redo just one stage

Stages (each is one of the scripts in this folder, run in its own process):
  world      generate the small world                     data/worlds/small/
  baselines  tune and score the baseline recommenders     reports/small/baselines.json
  twin       fit and score the shopper twin               reports/small/twin.json and models/small/twin_day*.pt
  whatif     ask the twin five what-if questions          reports/small/whatif.json
  evaluate   score the twin against the true store         reports/small/evaluation.json
A stage counts as done when all its outputs exist. The reports come with the repository but the world and the
fitted twin do not, so on a fresh copy the world and twin stages run. --force redoes the selected stages anyway.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PRESET = "small"
WORLD = Path("data/worlds") / PRESET
REPORTS = Path("reports") / PRESET
MODELS = Path("models") / PRESET

# name, command, test that the stage's outputs exist
STAGES = [
    ("world", ["scripts/generate_world.py", "--preset", PRESET], lambda: (WORLD / "manifest.json").exists()),
    ("baselines", ["scripts/run_baselines.py", "--world", str(WORLD)], lambda: (REPORTS / "baselines.json").exists()),
    ("twin", ["scripts/run_twin.py", "--world", str(WORLD)],
     lambda: (REPORTS / "twin.json").exists() and any(MODELS.glob("twin_day*.pt"))),
    ("whatif", ["scripts/run_whatif.py", "--world", str(WORLD)], lambda: (REPORTS / "whatif.json").exists()),
    ("evaluate", ["scripts/evaluate_twin.py", "--world", str(WORLD)],
     lambda: (REPORTS / "evaluation.json").exists()),
]


def main() -> None:
    names = [name for name, _, _ in STAGES]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=names, help="redo this stage and every stage after it")
    parser.add_argument("--only", choices=names, help="run just this stage")
    parser.add_argument("--force", action="store_true", help="redo stages even if their outputs exist")
    args = parser.parse_args()

    if args.only:
        selected = [args.only]
    elif args.start:
        selected = names[names.index(args.start):]
    else:
        selected = names
    t_all = time.perf_counter()
    for name, command, done in STAGES:
        if name not in selected:
            continue
        # a stage after --from reads what the earlier stages just rewrote, so it is redone too
        if done() and not (args.force or args.start):
            print(f"[{name}] already done, skipping")
            continue
        print(f"[{name}] running: python {' '.join(command)}", flush=True)
        t0 = time.perf_counter()
        result = subprocess.run([sys.executable, *command], check=False)
        if result.returncode != 0:
            sys.exit(f"[{name}] failed (exit code {result.returncode}). Fix the error above, then run "
                     f"python scripts/run_all.py --from {name}")
        print(f"[{name}] finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    print(f"All done in {(time.perf_counter() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
