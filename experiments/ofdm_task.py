"""A small OFDM channel-estimation task used by the leakage, seed and baseline experiments.

64 subcarriers, a pilot on every 4th one (16 pilots), an 8-tap Rayleigh channel with an
exponential power-delay profile. Input: least-squares estimates at the pilots; target: the
channel on all 64 subcarriers. For this Gaussian channel the LMMSE estimator is the best
possible (it is the MMSE estimator), so no learned estimator can beat it on fresh data.
"""
from __future__ import annotations

import numpy as np

N, PSTEP, L, DECAY = 64, 4, 8, 0.5
PILOTS = np.arange(0, N, PSTEP)
PDP = np.exp(-DECAY * np.arange(L))
PDP /= PDP.sum()
F = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(L)) / N)
R = (F * PDP) @ F.conj().T                         # channel covariance across subcarriers


def channels(g: int, rng, pdp=PDP) -> np.ndarray:
    taps = (rng.standard_normal((g, L)) + 1j * rng.standard_normal((g, L))) * np.sqrt(pdp / 2)
    return taps @ F.T


def observe(H: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """Noisy LS estimates at the pilots (unit-power pilots)."""
    s2 = 10 ** (-snr_db / 10)
    Hp = H[:, PILOTS]
    return Hp + np.sqrt(s2 / 2) * (rng.standard_normal(Hp.shape) + 1j * rng.standard_normal(Hp.shape))


def lmmse(Yp: np.ndarray, snr_db: float, Rm: np.ndarray = R) -> np.ndarray:
    s2 = 10 ** (-snr_db / 10)
    Rhp = Rm[:, PILOTS]
    Rpp = Rm[np.ix_(PILOTS, PILOTS)]
    W = Rhp @ np.linalg.inv(Rpp + s2 * np.eye(len(PILOTS)))
    return Yp @ W.T


def ls_interp(Yp: np.ndarray) -> np.ndarray:
    """Least squares at the pilots, linear interpolation in between (edge: hold)."""
    out = np.empty((Yp.shape[0], N), complex)
    k = np.arange(N)
    for i in range(Yp.shape[0]):
        out[i] = np.interp(k, PILOTS, Yp[i].real) + 1j * np.interp(k, PILOTS, Yp[i].imag)
    return out


def to_real(z: np.ndarray) -> np.ndarray:
    return np.hstack([z.real, z.imag])


def to_complex(x: np.ndarray) -> np.ndarray:
    h = x.shape[1] // 2
    return x[:, :h] + 1j * x[:, h:]


def nmse_db(Hhat: np.ndarray, H: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(np.abs(Hhat - H) ** 2) / np.sum(np.abs(H) ** 2)))


def dataset(groups: int, per_group: int, snr_db: float, seed: int):
    """`groups` channel realizations, `per_group` noisy observations of each."""
    rng = np.random.default_rng(seed)
    H = channels(groups, rng)
    Hrep = np.repeat(H, per_group, axis=0)
    Yp = observe(Hrep, snr_db, rng)
    g = np.repeat(np.arange(groups), per_group)
    return Yp, Hrep, g
