from __future__ import annotations

"""Pipeline entry point.

The run is organised into seven stages.  ``--mode all`` executes them in
order; any other mode executes the single stage it names.  Stage boundaries,
headers, timings and progress bars are presentation only -- every stage calls
exactly the same functions with exactly the same arguments as before, so the
numerical results are unaffected.
"""

import argparse
import json
import time
from pathlib import Path

from src.evaluation.experiments import run_all_experiments
from src.evaluation.pipeline import prepare_project, train_models, generate_synthetic_dataset
from src.utils.progress import pipeline_header, pipeline_summary, stage
from src.visualization.plots import make_all_plots

TITLE = 'Spacecraft Hybrid PINN + UKF + MEKF Benchmark'

MODES = ['synth', 'train', 'evaluate', 'plot', 'theory', 'storm', 'ablate', 'all']

# (mode, stage name).  The order here is the execution order for --mode all
# and fixes the "Stage i/7" numbering used in the headers.
STAGES = [
    ('synth', 'Synthetic Data Generation'),
    ('train', 'Training'),
    ('evaluate', 'Evaluation'),
    ('theory', 'Theory Analysis'),
    ('plot', 'Plot Generation'),
    ('storm', 'Storm Experiment'),
    ('ablate', 'Ablation'),
]
N_STAGES = len(STAGES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Spacecraft hybrid PINN + UKF + MEKF benchmark')
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--mode', type=str, default='all', choices=MODES,
                   help='Single pipeline stage to run, or "all" for the full seven-stage run.')
    return p.parse_args()


# --------------------------------------------------------------------------
# Stage bodies.  Each is exactly the call sequence the previous main.py used.
# --------------------------------------------------------------------------

def _stage_synth(project) -> None:
    generate_synthetic_dataset(project)


def _stage_train(project) -> None:
    train_models(project)


def _stage_evaluate(project) -> None:
    results = run_all_experiments(project)
    (project.output_dir / 'results.json').write_text(json.dumps(results, indent=2))


def _stage_theory(project) -> None:
    from src.evaluation.theory import run_theory_analysis
    run_theory_analysis(project)


def _stage_plot(project) -> None:
    make_all_plots(project)


def _stage_storm(project) -> None:
    from src.evaluation.storm_experiment import run_storm_experiment
    from src.visualization.plots import make_storm_plots
    run_storm_experiment(project)
    make_storm_plots(project)


def _stage_ablate(project) -> None:
    from src.evaluation.ablation import run_ablation
    run_ablation(project)


STAGE_FUNCS = {
    'synth': _stage_synth,
    'train': _stage_train,
    'evaluate': _stage_evaluate,
    'theory': _stage_theory,
    'plot': _stage_plot,
    'storm': _stage_storm,
    'ablate': _stage_ablate,
}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    project = prepare_project(config_path)

    pipeline_header(TITLE, config_path, args.mode, project.output_dir)

    selected = [(i, mode, name) for i, (mode, name) in enumerate(STAGES, start=1)
                if args.mode in {'all', mode}]

    t0 = time.perf_counter()
    completed = 0
    for index, mode, name in selected:
        with stage(index, N_STAGES, name):
            STAGE_FUNCS[mode](project)
        completed += 1

    pipeline_summary(completed, len(selected), time.perf_counter() - t0,
                     project.output_dir)


if __name__ == '__main__':
    main()
