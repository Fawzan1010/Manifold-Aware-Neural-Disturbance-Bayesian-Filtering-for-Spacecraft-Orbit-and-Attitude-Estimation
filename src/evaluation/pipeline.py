from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

from src.dynamics.simulator import generate_dataset, Trajectory, build_params
from src.dynamics.spacecraft import SpacecraftParams
from src.utils.progress import progress_bar, section
from src.utils.reproducibility import set_seed, ensure_dir


@dataclass
class Project:
    root: Path
    config: dict
    output_dir: Path
    data_dir: Path
    model_dir: Path
    figure_dir: Path
    table_dir: Path
    storm_dir: Path | None = None

    def spacecraft_params(self, atmosphere_override: str | None = None):
        """Single construction point for SpacecraftParams.

        Building the dataclass in each call site would mean the atmosphere
        model had to be attached in each of them too, and any site that forgot
        would silently fall back to the exponential profile.
        """
        return build_params(self.config, atmosphere_override=atmosphere_override)


def prepare_project(config_path: Path) -> Project:
    raw = yaml.safe_load(config_path.read_text())
    cfg = dict(raw)
    proj = cfg.pop('project', {})
    cfg.update(proj)
    root = config_path.parent.parent
    output_dir = ensure_dir(Path(cfg['output_dir']))
    data_dir = ensure_dir(output_dir / 'data')
    model_dir = ensure_dir(output_dir / 'models')
    figure_dir = ensure_dir(output_dir / 'figures')
    table_dir = ensure_dir(output_dir / 'tables')
    storm_dir = ensure_dir(output_dir / 'storm')
    set_seed(int(cfg['seed']))
    return Project(root, cfg, output_dir, data_dir, model_dir, figure_dir,
                   table_dir, storm_dir)


def _load_split(path: Path) -> list[Trajectory]:
    data = np.load(path, allow_pickle=True)
    return list(data['trajectories'])


def generate_synthetic_dataset(project: Project) -> None:
    cfg = project.config
    synth = cfg['synthetic']
    section('Splits: train, val, test')
    generate_dataset(project.data_dir, cfg, 'train', int(synth['train_trajectories']), int(cfg['seed']))
    generate_dataset(project.data_dir, cfg, 'val', int(synth['val_trajectories']), int(cfg['seed']) + 1000)
    generate_dataset(project.data_dir, cfg, 'test', int(synth['test_trajectories']), int(cfg['seed']) + 2000)
    section('Collecting reproducibility metadata')
    meta = {'config': cfg, 'environment': _environment_metadata(cfg)}
    (project.output_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str))


def train_models(project: Project):
    # Imported here so that dataset generation and the classical filters do
    # not require torch to be installed.
    from src.pinn.train import train_pinn
    from src.models.train import train_transformer

    with progress_bar('Loading splits', total=2, unit='split') as bar:
        train = _load_split(project.data_dir / 'train.npz')
        bar.update(1)
        val = _load_split(project.data_dir / 'val.npz')
        bar.update(1)
    section('Model 1/2: PINN residual network')
    pinn = train_pinn(train, val, project.config, project.model_dir, project.config['device'])
    section('Model 2/2: Transformer baseline')
    transformer = train_transformer(train, val, project.config, project.model_dir, project.config['device'])
    return {'pinn': pinn, 'transformer': transformer}


def _environment_metadata(cfg: dict) -> dict:
    """Versions of the packages whose numerical output enters the results."""
    import platform
    import numpy as _np
    import scipy as _sp

    meta = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'numpy': _np.__version__,
        'scipy': _sp.__version__,
    }
    try:
        import pymsis
        meta['pymsis'] = getattr(pymsis, '__version__', 'unknown')
    except Exception:
        meta['pymsis'] = None
    try:
        import torch
        meta['torch'] = torch.__version__
    except Exception:
        meta['torch'] = None
    try:
        from src.dynamics.atmosphere import make_atmosphere
        meta['atmosphere'] = make_atmosphere(cfg).describe()
    except Exception as exc:
        meta['atmosphere'] = {'error': str(exc)}
    return meta
