from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class MissionDataLoader(ABC):
    @abstractmethod
    def load(self, path: str | Path) -> pd.DataFrame:
        raise NotImplementedError


class GRACEFOLoader(MissionDataLoader):
    def load(self, path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'GRACE-FO-like file not found: {p}')
        return pd.read_csv(p)


class SwarmLoader(MissionDataLoader):
    def load(self, path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Swarm-like file not found: {p}')
        return pd.read_csv(p)


class GenericTelemetryLoader(MissionDataLoader):
    def load(self, path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Telemetry file not found: {p}')
        suffix = p.suffix.lower()
        if suffix in {'.csv', '.txt'}:
            return pd.read_csv(p)
        if suffix in {'.parquet'}:
            return pd.read_parquet(p)
        raise ValueError(f'Unsupported telemetry format: {suffix}')
