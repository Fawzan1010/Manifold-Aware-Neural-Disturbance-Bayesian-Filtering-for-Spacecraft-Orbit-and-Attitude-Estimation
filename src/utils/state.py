from __future__ import annotations

import numpy as np
from dataclasses import dataclass

Q = slice(0, 4)
W = slice(4, 7)
R = slice(7, 10)
V = slice(10, 13)
BG = slice(13, 16)
BA = slice(16, 19)
TD = slice(19, 22)
AD = slice(22, 25)
STATE_DIM = 25
ERROR_DIM = 24


@dataclass
class SpacecraftState:
    q: np.ndarray
    w: np.ndarray
    r: np.ndarray
    v: np.ndarray
    bg: np.ndarray
    ba: np.ndarray
    td: np.ndarray
    ad: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.hstack([self.q, self.w, self.r, self.v, self.bg, self.ba, self.td, self.ad])

    @staticmethod
    def from_vector(x: np.ndarray) -> 'SpacecraftState':
        x = np.asarray(x, dtype=float).reshape(STATE_DIM)
        return SpacecraftState(x[Q], x[W], x[R], x[V], x[BG], x[BA], x[TD], x[AD])


def pack_state(q, w, r, v, bg, ba, td, ad) -> np.ndarray:
    return np.hstack([q, w, r, v, bg, ba, td, ad]).astype(float)


def unpack_state(x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(STATE_DIM)
    return {'q': x[Q], 'w': x[W], 'r': x[R], 'v': x[V], 'bg': x[BG], 'ba': x[BA], 'td': x[TD], 'ad': x[AD]}
