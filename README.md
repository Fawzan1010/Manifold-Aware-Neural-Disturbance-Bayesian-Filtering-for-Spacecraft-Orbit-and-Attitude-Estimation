# Spacecraft Hybrid PINN + UKF + MEKF Benchmark

This project provides a fully runnable synthetic research benchmark for spacecraft orbit-and-attitude estimation under disturbances, bias drift, maneuvers, degraded sensors, and storm-like conditions.

## What is included
- Classical filters: EKF, UKF, MEKF
- Learning-only baselines: PINN-only, Transformer-only
- Hybrid methods: PINN+EKF, PINN+UKF, PINN+MEKF, Transformer+MEKF, PINN+UKF+MEKF
- Synthetic spacecraft simulator with quaternion attitude, orbit propagation, sensor models, bias and disturbance states
- Monte Carlo evaluation, metrics tables, LaTeX export, and publication-style plots

## Requirements
Python 3.10+ recommended.

```bat
pip install -r requirements.txt
```

## Run on Linux
```bash
python main.py --config configs/default.yaml --mode all
```

## Run on Windows Command Prompt
```bat
python main.py --config configs\default.yaml --mode all
```

## Run on Windows PowerShell
```powershell
python .\main.py --config .\configs\default.yaml --mode all
```

## Pipeline stages
`--mode all` runs seven stages in order, each with a header, a tqdm progress
bar and an elapsed-time footer:

| Stage | Mode | What it does |
|-------|------|--------------|
| 1 | `synth`    | Synthetic dataset generation |
| 2 | `train`    | PINN and Transformer training |
| 3 | `evaluate` | Accuracy and runtime benchmark |
| 4 | `theory`   | Observability / consistency / bounded-error analysis |
| 5 | `plot`     | Figure generation |
| 6 | `storm`    | 17 March 2015 storm experiment |
| 7 | `ablate`   | Hyperparameter ablation sweep |

Any single stage can be run on its own with `--mode <stage>`.  Stages 6 and 7
are the expensive ones; run them separately if you only need stages 1-5.

## Staged workflow
Generate synthetic data only:
```bash
python main.py --mode synth
```

Train the PINN and Transformer baselines:
```bash
python main.py --mode train
```

Run the benchmark comparison table:
```bash
python main.py --mode evaluate
```

Generate figures:
```bash
python main.py --mode plot
```

## Outputs
- `outputs/data/`: synthetic train/val/test splits
- `outputs/models/`: trained PINN and Transformer checkpoints
- `outputs/tables/`: CSV and LaTeX summaries
- `outputs/figures/`: PDF figures
- `outputs/results.json`: experiment summaries
- `outputs/metadata.json`: reproducibility metadata

## Notes
The code is designed to be extensible for future integration with GRACE-FO-like, Swarm-like, and generic telemetry loaders.
