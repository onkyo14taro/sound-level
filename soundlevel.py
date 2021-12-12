"""Module for the frequency weightings of the sound pressure level.

The standards for the weightings are defined by ANSI [1] and IEC [2].
The digital filters designed by bilinear transform with prewarping is
introduced by [3].

References
----------
[1] American National Standards Institute, “ANSI S1.43:
    Specifications for Integrating-averaging Sound Level Meters,”
    Acoustical Society of America, 1997.
[2] International Electrotechnical Commission, “Electroacoustics:
    Sound Level Meters. Specifications. Part 1,”
    International Electrotechnical Commission, 2013.
[3] A. N. Rimell, N. J. Mansfield, and G. S. Paddan,
    “Design of digital filters for frequency weightings (A and C)
    required for risk assessments of workers exposed to noise,”
    Ind. Health, pp. 3–2013, 2014.
"""
from collections.abc import Sized
from functools import lru_cache
from typing import Optional, Union, Sequence, Tuple
import warnings

import librosa
import numpy as np
import pandas as pd
from scipy import signal


################################################################################
################################################################################
### Constant values
################################################################################
################################################################################
TOLERANCE = pd.DataFrame(
    [
        [    10,   -70.4, -38.2, -14.3, 0.0, +3.0, float('-inf'), +5.0, float('-inf'), +2.0, -5.0, +4.0, -4.0, +5.0, float('-inf')],
        [    12.5, -63.4, -33.2, -11.2, 0.0, +2.5, float('-inf'), +5.0, float('-inf'), +2.0, -4.0, +3.5, -3.5, +5.0, float('-inf')],
        [    16,   -56.7, -28.5,  -8.5, 0.0, +2.0,  -4.0,         +5.0, float('-inf'), +2.0, -3.0, +3.0, -3.0, +5.0, float('-inf')],
        [    20,   -50.5, -24.2,  -6.2, 0.0, +2.0,  -2.0,         +3.0, -3.0,          +2.0, -2.0, +2.5, -2.5, +3.0, -3.0],
        [    25,   -44.7, -20.4,  -4.4, 0.0, +2.0,  -1.5,         +3.0, -3.0,          +1.5, -1.5, +2.0, -2.0, +3.0, -3.0],
        [    31.5, -39.4, -17.1,  -3.0, 0.0, +1.5,  -1.5,         +3.0, -3.0,          +1.0, -1.0, +1.5, -1.5, +3.0, -3.0],
        [    40,   -34.6, -14.2,  -2.0, 0.0, +1.0,  -1.0,         +2.0, -2.0,          +1.0, -1.0, +1.5, -1.5, +2.0, -2.0],
        [    50,   -30.2, -11.6,  -1.3, 0.0, +1.0,  -1.0,         +2.0, -2.0,          +1.0, -1.0, +1.0, -1.0, +2.0, -2.0],
        [    63,   -26.2,  -9.3,  -0.8, 0.0, +1.0,  -1.0,         +2.0, -2.0,          +1.0, -1.0, +1.0, -1.0, +2.0, -2.0],
        [    80,   -22.5,  -7.4,  -0.5, 0.0, +1.0,  -1.0,         +2.0, -2.0,          +1.0, -1.0, +1.0, -1.0, +1.5, -1.5],
        [   100,   -19.1,  -5.6,  -0.3, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [   125,   -16.1,  -4.2,  -0.2, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [   160,   -13.4,  -3.0,  -0.1, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [   200,   -10.9,  -2.0,   0.0, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [   250,    -8.6,  -1.3,   0.0, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [   315,    -6.6,  -0.8,   0.0, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [   400,    -4.8,  -0.5,   0.0, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [   500,    -3.2,  -0.3,   0.0, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [   630,    -1.9,  -0.1,   0.0, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [   800,    -0.8,   0.0,   0.0, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [ 1_000,     0.0,   0.0,   0.0, 0.0, +0.7,  -0.7,         +1.0, -1.0,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [ 1_250,    +0.6,   0.0,   0.0, 0.0, +1.0,  -1.0,         +1.5, -1.5,          +0.7, -0.7, +1.0, -1.0, +1.5, -1.5],
        [ 1_600,    +1.0,   0.0,  -0.1, 0.0, +1.0,  -1.0,         +2.0, -2.0,          +0.7, -0.7, +1.0, -1.0, +2.0, -2.0],
        [ 2_000,    +1.2,  -0.1,  -0.2, 0.0, +1.0,  -1.0,         +2.0, -2.0,          +0.7, -0.7, +1.0, -1.0, +2.0, -2.0],
        [ 2_500,    +1.3,  -0.2,  -0.3, 0.0, +1.0,  -1.0,         +2.5, -2.5,          +0.7, -0.7, +1.0, -1.0, +2.5, -2.5],
        [ 3_150,    +1.2,  -0.4,  -0.5, 0.0, +1.0,  -1.0,         +2.5, -2.5,          +0.7, -0.7, +1.0, -1.0, +2.5, -2.5],
        [ 4_000,    +1.0,  -0.7,  -0.8, 0.0, +1.0,  -1.0,         +3.0, -3.0,          +0.7, -0.7, +1.0, -1.0, +3.0, -3.0],
        [ 5_000,    +0.5,  -1.2,  -1.3, 0.0, +1.5,  -1.5,         +3.5, -3.5,          +1.0, -1.0, +1.5, -1.5, +3.5, -3.5],
        [ 6_300,    -0.1,  -1.9,  -2.0, 0.0, +1.5,  -2.0,         +4.5, -4.5,          +1.0, -1.5, +1.5, -2.0, +4.5, -4.5],
        [ 8_000,    -1.1,  -2.9,  -3.0, 0.0, +1.5,  -2.5,         +5.0, -5.0,          +1.0, -2.0, +1.5, -3.0, +5.0, -5.0],
        [10_000,    -2.5,  -4.3,  -4.4, 0.0, +2.0,  -3.0,         +5.0, float('-inf'), +2.0, -3.0, +2.0, -4.0, +5.0, float('-inf')],
        [12_500,    -4.3,  -6.1,  -6.2, 0.0, +2.0,  -5.0,         +5.0, float('-inf'), +2.0, -3.0, +3.0, -6.0, +5.0, float('-inf')],
        [16_000,    -6.6,  -8.4,  -8.5, 0.0, +2.5, -16.0,         +5.0, float('-inf'), +2.0, -3.0, +3.0, float('-inf'), +5.0, float('-inf')],
        [20_000,    -9.3, -11.1, -11.2, 0.0, +3.0, float('-inf'), +5.0, float('-inf'), +2.0, -3.0, +3.0, float('-inf'), +5.0, float('-inf')],
    ],
    columns=[
        'Frequency', 'A', 'B', 'C', 'Z',
        'IEC1_upper', 'IEC1_lower',
        'IEC2_upper', 'IEC2_lower',
        'ANSI0_upper', 'ANSI0_lower',
        'ANSI1_upper', 'ANSI1_lower',
        'ANSI2_upper', 'ANSI2_lower',
    ]
)
FREQUENCY = TOLERANCE['Frequency'].values
FREQUENCY.setflags(write=False)


################################################################################
################################################################################
### Helper functions
################################################################################
################################################################################
def abs2(x: np.ndarray) -> np.ndarray:
    """Calculate the squared absolute value of a given array.

    Parameters
    ----------
    x : numpy.ndarray
        Wave.

    Returns
    -------
    x_abs2 : numpy.ndarray
        The squared absolute value of `x`.
    """
    if np.issubdtype(x.dtype, np.complexfloating):
        return x.real**2 + x.imag**2
    return x**2


def calc_db(x: np.ndarray, min_db: Optional[float] = None) -> np.ndarray:
    """Calculate the decibel value of a given array.

    Parameters
    ----------
    x : numpy.ndarray
        Wave.
    min_db : float, optional
        Minimum threshold [dB].
        If not `None`, values less than `min_db` will be clipped to `min_db`.
        By default, `None`.

    Returns
    -------
    x_db : numpy.ndarray
        The decibel value of `x`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        db = 10*np.log10(abs2(x))
    if min_db is not None:
        db[db < min_db] = min_db
    return db


def frame_wave(
    x: np.ndarray,
    fs: float,
    sec: float = 1.0,
) -> np.ndarray:
    """Split a wave into frames.

    The length of each frame is `round(fs * sec)` [sample].
    Trailing edge that is not divisible by
    `round(fs * sec)` will be truncated.

    Parameters
    ----------
    x : numpy.ndarray
        Wave.
    fs : float
        Sampling frequency [Hz].
    sec : float
        Frame width [second].
        By default, `1.0`.

    Returns
    -------
    x_frame : np.ndarray [shape=(n_frames, round(fs*sec))]
        The frames.
    """
    assert x.ndim == 1, 'x.ndim must be 1.'
    assert fs > 0.0, 'fs must be greater than 0.0.'
    assert sec > 0.0, 'sec must be greater than 0.0.'
    sample = round(fs * sec)
    length = x.shape[0]
    length -= length % sample
    return x[:length].reshape(-1, sample)


################################################################################
################################################################################
### Fiilter parameters
################################################################################
################################################################################
@lru_cache()
def zpk_s_weighting(
    weighting: str = 'A',
    *,
    fs: float = 96_000.0,
    prewarping: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return the zeros, poles and gain of the weightings (analogue).

    The parameters for the weightings are referred to [1].

    Parameters
    ----------
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either `'A'`, `'B'`, `'C'`, or `'Z'`.
        By default, `'A'`.
    fs : float
        Sampling frequency [Hz].
        This parameter is used only when `prewarping == True`.
        By default, `96_000.0`.
    prewarping : bool
        Whether to apply prewarping for the bilinear transform.
        If `False`, this function returns the analogue filter prameters.
        By default, `False`.

    Returns
    -------
    zeros : np.ndarray
    poles : np.ndarray
    gain : np.float64

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    """
    f = np.array([
           20.598997,
          107.65265,
          737.86223,
        12194.22,
          158.48932
    ])
    w = f * (2*np.pi)
    if prewarping:
        w = (2*fs)*np.tan(f/fs*np.pi)
    else:
        w = f * (2*np.pi)
    G_A = 10 ** (  1.9997 / 20.0)
    G_B = 10 ** ( -0.1696 / 20.0)
    G_C = 10 ** ( -0.0619 / 20.0)
    if weighting == 'A':
        zeros = np.array([0.0, 0.0, 0.0, 0.0])
        poles = np.array([-w[0], -w[0], -w[1], -w[2], -w[3], -w[3]])
        gain = G_A * w[3]**2
    elif weighting == 'B':
        zeros = np.array([0.0, 0.0, 0.0])
        poles = np.array([-w[0], -w[0],               -w[3], -w[3], -w[4]])
        gain = G_B * w[3]**2
    elif weighting == 'C':
        zeros = np.array([0.0, 0.0])
        poles = np.array([-w[0], -w[0],               -w[3], -w[3]])
        gain = G_C * w[3]**2
    elif weighting == 'Z':
        zeros = np.array([])
        poles = np.array([])
        gain = 1
    else:
        raise ValueError(
            f"weighting must be either 'A', 'B', 'C', or 'Z'; found {weighting}"
        )
    zeros.setflags(write=False)
    poles.setflags(write=False)
    return zeros, poles, gain


@lru_cache()
def zpk_z_weighting(
    weighting: str = 'A',
    fs: float = 96_000.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return the zeros, poles and gain of weightings (digital).

    The parameters for the weightings are referred to [1].
    The digital filter is designed by the bilinear transform [2].

    Parameters
    ----------
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either `'A'`, `'B'`, `'C'`, or `'Z'`.
        By default `'A'`.
    fs : float
        Sampling frequency [Hz].
        By default `96_000.0`.

    Returns
    -------
    zeros : np.ndarray
    poles : np.ndarray
    gain : np.float64

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    [2] A. N. Rimell, N. J. Mansfield, and G. S. Paddan,
        “Design of digital filters for frequency weightings (A and C)
        required for risk assessments of workers exposed to noise,”
        Ind. Health, pp. 3–2013, 2014.
    """
    zpk_s = zpk_s_weighting(weighting, fs=fs, prewarping=True)
    zeros, poles, gain = signal.bilinear_zpk(*zpk_s, fs)
    zeros.setflags(write=False)
    poles.setflags(write=False)
    return zeros, poles, gain


@lru_cache()
def sos_weighting(
    weighting: str = 'A',
    fs: float = 96_000.0,
) -> np.ndarray:
    """Return the series of second-order sections of weightings.

    The parameters for the weightings are referred to [1].
    The digital filter is designed by the bilinear transform [2].

    Parameters
    ----------
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either `'A'`, `'B'`, `'C'`, or `'Z'`.
        By default `'A'`.
    fs : float
        Sampling frequency [Hz].
        By default `96_000.0`.

    Returns
    -------
    sos : np.ndarray
        Series of second-order sections.

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    [2] A. N. Rimell, N. J. Mansfield, and G. S. Paddan,
        “Design of digital filters for frequency weightings (A and C)
        required for risk assessments of workers exposed to noise,”
        Ind. Health, pp. 3–2013, 2014.
    """
    zpk_z = zpk_z_weighting(weighting, fs)
    return signal.zpk2sos(*zpk_z)


################################################################################
################################################################################
### Validation
################################################################################
################################################################################
@lru_cache()
def torelance_standard(
    weighting: str = 'A',
    standard: str = 'ANSI0',
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the tolerance for ANSI or IEC.

    Parameters
    ----------
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either 'A', 'B', 'C'`, or `'Z'`.
        By default, `'A'`.
    standard : str
        Standard referenced.
        Choose one of the following:
        - `'ANSI0'`: ANSI S.1-43 Type 0 [1]
        - `'ANSI1'`: ANSI S.1-43 Type 1 [1]
        - `'ANSI2'`: ANSI S.1-43 Type 2 [1]
        - `'IEC1'`: IEC 61672-1 Class 1 [2]
        - `'IEC2'`: IEC 61672-1 Class 2 [2]
        By default, `'ANSI0'` (the strictest standard).

    Returns
    -------
    freq : numpy.ndarray [shape=(34, )]
        Frequencies.
    tol : numpy.ndarray [shape=(34, 2)]
        The decibel value of `x`.
        `tol[:, 0]` is the upper bound and
        `tol[:, 1]` is the lower bound.
        `tol[f]` corresponds to `freq[f]`.

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    [2] International Electrotechnical Commission, “Electroacoustics:
        Sound Level Meters. Specifications. Part 1,”
        International Electrotechnical Commission, 2013.
    """
    if weighting not in {'A', 'B', 'C', 'Z'}:
        raise ValueError()
    if standard not in {'IEC1', 'IEC2', 'ANSI0', 'ANSI1', 'ANSI2'}:
        raise ValueError()
    db_center = TOLERANCE[weighting].values[:, None]
    db_limit  = TOLERANCE[[f'{standard}_upper', f'{standard}_lower']].values
    tol = db_center + db_limit
    return FREQUENCY, tol


@lru_cache()
def satisfied_standard(
    fs: float,
    weighting: str = 'A',
    standard: str = 'ANSI0',
) -> bool:
    """Return whether `standard` for `weighting` can be
    satisfied by a sampling frequency `fs`.

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz].
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either 'A', 'B', 'C'`, or `'Z'`.
        By default, `'A'`.
    standard : str
        Standard referenced.
        Choose one of the following:
        - `'ANSI0'`: ANSI S.1-43 Type 0 [1]
        - `'ANSI1'`: ANSI S.1-43 Type 1 [1]
        - `'ANSI2'`: ANSI S.1-43 Type 2 [1]
        - `'IEC1'`: IEC 61672-1 Class 1 [2]
        - `'IEC2'`: IEC 61672-1 Class 2 [2]
        By default, `'ANSI0'` (the strictest standard).

    Returns
    -------
    satisfied : bool
        Whether `fs` satisfies `standard` for `weighting`.

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    [2] International Electrotechnical Commission, “Electroacoustics:
        Sound Level Meters. Specifications. Part 1,”
        International Electrotechnical Commission, 2013.
    """
    f, tol = torelance_standard(weighting, standard)
    f = f / fs          # normalized frequency
    worN = (2*np.pi)*f  # normalized angular frequency
    _, h = signal.freqz_zpk(*zpk_z_weighting(weighting, fs), worN=worN)
    # The frequency response above the Nyquist frequency is regarded as -inf dB.
    h[f >= 0.5] = 0.0
    h_db = calc_db(h)
    return np.all((tol[:, 1] <= h_db) & (h_db <= tol[:, 0]))


@lru_cache()
def minimum_fs(
    weighting: str = 'A',
    standard: str = 'ANSI0',
) -> float:
    """Return the minimum sampling frequency
    to satisfy `standard` for `weighting`.

    Parameters
    ----------
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either 'A', 'B', 'C'`, or `'Z'`.
        By default, `'A'`.
    standard : str
        Standard referenced.
        Choose one of the following:
        - `'ANSI0'`: ANSI S.1-43 Type 0 [1]
        - `'ANSI1'`: ANSI S.1-43 Type 1 [1]
        - `'ANSI2'`: ANSI S.1-43 Type 2 [1]
        - `'IEC1'`: IEC 61672-1 Class 1 [2]
        - `'IEC2'`: IEC 61672-1 Class 2 [2]
        By default, `'ANSI0'` (the strictest standard).

    Returns
    -------
    fs_min : float
        The minimum sampling frequency to satisfy `standard` for `weighting`.

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    [2] International Electrotechnical Commission, “Electroacoustics:
        Sound Level Meters. Specifications. Part 1,”
        International Electrotechnical Commission, 2013.
    """
    fs = 10_000.0
    while True:
        if satisfied_standard(fs, weighting, standard):
            return fs
        fs += 100.0


def validate_standard(
    fs: float,
    weighting: str = 'A',
    standard: str = 'ANSI0',
    raise_error: bool = False,
) -> bool:
    """Return whether `standard` for `weighting` can be satisfied
    by a sampling frequency `fs`.

    This function is the same as `satisfied_standard()`,
    except that it raise an error or a warning
    if `standard` is not satisfied.

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz].
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either 'A', 'B', 'C'`, or `'Z'`.
        By default, `'A'`.
    standard : str
        Standard referenced.
        Choose one of the following:
        - `'ANSI0'`: ANSI S.1-43 Type 0 [1]
        - `'ANSI1'`: ANSI S.1-43 Type 1 [1]
        - `'ANSI2'`: ANSI S.1-43 Type 2 [1]
        - `'IEC1'`: IEC 61672-1 Class 1 [2]
        - `'IEC2'`: IEC 61672-1 Class 2 [2]
        By default, `'ANSI0'` (the strictest standard).
    raise_error : bool
        If `True`, raise a `ValueError` when `standard` is not satisfied.
        If `False`, throw a warning when `standard` is not satisfied.
        By default, `False`.

    Returns
    -------
    satisfied : bool
        Whether `fs` satisfies `standard` for `weighting`.

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    [2] International Electrotechnical Commission, “Electroacoustics:
        Sound Level Meters. Specifications. Part 1,”
        International Electrotechnical Commission, 2013.
    """
    satisfied = satisfied_standard(fs, weighting, standard)
    if not satisfied:
        fs_required = minimum_fs(weighting, standard)
        msg = \
            f'Criterion {standard} is not satisfied at fs={fs} Hz. ' \
            f'To satisfy it, fs must be >= {fs_required} Hz.'
        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    return satisfied


################################################################################
################################################################################
### Equivalent continuous sound level and percentile sound pressure level
################################################################################
################################################################################
def weight_wave(
    wave: np.ndarray,
    fs_orig: float,
    *,
    weighting: str = 'A',
    fs_weighting: float = 96_000.0,
    standard: str = 'ANSI0',
    must_satisfy_standard: bool = True,
) -> np.ndarray:
    """Filter a wave by a digital filter of a weighting.

    The parameters for the weightings are referred to [1].
    The digital filter is designed by the bilinear transform [2].

    Parameters
    ----------
    wave : numpy.ndarray [shape=(wave_len, )]
        Wave.
    fs_orig : float
        Sampling frequency [Hz] of `wave`.
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either `'A'`, `'B'`, `'C'`, or `'Z'`.
        By default `'A'`.
    fs_weighting : float
        Sampling frequency [Hz] for digital-filtering.
        By default `96_000.0`.
    standard : str
        Standard referenced.
        If `fs_weighting` does not satisfy `standard`,
        raise a warning (if `must_satisfy_standard == False`)
        or an error (if `must_satisfy_standard == True`).
        Choose one of the following:
        - `'ANSI0'`: ANSI S.1-43 Type 0 [1]
        - `'ANSI1'`: ANSI S.1-43 Type 1 [1]
        - `'ANSI2'`: ANSI S.1-43 Type 2 [1]
        - `'IEC1'`: IEC 61672-1 Class 1 [3]
        - `'IEC2'`: IEC 61672-1 Class 2 [3]
        By default, `'ANSI0'` (the strictest standard).
    must_satisfy_standard : bool
        If `fs_weighting` does not satisfy `standard`,
        raise a warning (if `must_satisfy_standard == False`)
        or an error (if `must_satisfy_standard == True`).

    Returns
    -------
    wave_weighted : np.ndarray [shape=(wave_resampled_len, )]
        Weighted wave.

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    [2] A. N. Rimell, N. J. Mansfield, and G. S. Paddan,
        “Design of digital filters for frequency weightings (A and C)
        required for risk assessments of workers exposed to noise,”
        Ind. Health, pp. 3–2013, 2014.
    [3] International Electrotechnical Commission, “Electroacoustics:
        Sound Level Meters. Specifications. Part 1,”
        International Electrotechnical Commission, 2013.
    """
    validate_standard(fs_weighting, weighting, standard,
                      raise_error=must_satisfy_standard)
    wave = librosa.resample(wave, fs_orig, fs_weighting)
    if weighting == 'Z':
        return wave
    sos = sos_weighting(weighting, fs_weighting)
    return signal.sosfilt(sos, wave)


def equivalent_level(
    wave: np.ndarray,
    fs_orig: float,
    *,
    weighting: str = 'A',
    fs_weighting: float = 96_000.0,
    standard: str = 'ANSI0',
    must_satisfy_standard: bool = True,
) -> float:
    """Calculate equivalent continuous sound level (L_eq).

    The parameters for the weightings are referred to [1].
    The digital filter is designed by the bilinear transform [2].

    Parameters
    ----------
    wave : numpy.ndarray [shape=(wave_len, )]
        Audio wave.
    fs_orig : float
        Sampling frequency [Hz] of `wave`.
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either `'A'`, `'B'`, `'C'`, or `'Z'`.
        By default `'A'`.
    fs_weighting : float
        Sampling frequency [Hz] for digital-filtering.
        By default `96_000.0`.
    standard : str
        Standard referenced.
        If `fs_weighting` does not satisfy `standard`,
        raise a warning (if `must_satisfy_standard == False`)
        or an error (if `must_satisfy_standard == True`).
        Choose one of the following:
        - `'ANSI0'`: ANSI S.1-43 Type 0 [1]
        - `'ANSI1'`: ANSI S.1-43 Type 1 [1]
        - `'ANSI2'`: ANSI S.1-43 Type 2 [1]
        - `'IEC1'`: IEC 61672-1 Class 1 [3]
        - `'IEC2'`: IEC 61672-1 Class 2 [3]
        By default, `'ANSI0'` (the strictest standard).
    must_satisfy_standard : bool
        If `fs_weighting` does not satisfy `standard`,
        raise a warning (if `must_satisfy_standard == False`)
        or an error (if `must_satisfy_standard == True`).

    Returns
    -------
    L_eq : float
        Equivalent continuous sound level.

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    [2] A. N. Rimell, N. J. Mansfield, and G. S. Paddan,
        “Design of digital filters for frequency weightings (A and C)
        required for risk assessments of workers exposed to noise,”
        Ind. Health, pp. 3–2013, 2014.
    [3] International Electrotechnical Commission, “Electroacoustics:
        Sound Level Meters. Specifications. Part 1,”
        International Electrotechnical Commission, 2013.
    """
    wave = weight_wave(wave, fs_orig,
                       weighting=weighting,
                       fs_weighting=fs_weighting,
                       standard=standard,
                       must_satisfy_standard=must_satisfy_standard)
    return 10*np.log10(np.var(wave))


def percentile_level(
    wave: np.ndarray,
    fs_orig: float,
    x : Union[float, Sequence[float]] = (5, 50, 95),
    *,
    weighting: str = 'A',
    fs_weighting: float = 96_000.0,
    frame_sec: float = 1.0,
    standard: str = 'ANSI0',
    must_satisfy_standard: bool = True,
) -> Union[float, np.ndarray]:
    """Calculate percentile sound pressure level (L_x).

    The parameters for the weightings are referred to [1].
    The digital filter is designed by the bilinear transform [2].

    Parameters
    ----------
    wave : numpy.ndarray [shape=(wave_len, )]
        Audio wave.
    fs_orig : float
        Sampling frequency [Hz] of `wave`.
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either `'A'`, `'B'`, `'C'`, or `'Z'`.
        By default `'A'`.
    fs_weighting : float
        Sampling frequency [Hz] for digital-filtering.
        By default `96_000.0`.
    frame_sec : float
        Frame width [second].
        By default, `1.0`.
    standard : str
        Standard referenced.
        If `fs_weighting` does not satisfy `standard`,
        raise a warning (if `must_satisfy_standard == False`)
        or an error (if `must_satisfy_standard == True`).
        Choose one of the following:
        - `'ANSI0'`: ANSI S.1-43 Type 0 [1]
        - `'ANSI1'`: ANSI S.1-43 Type 1 [1]
        - `'ANSI2'`: ANSI S.1-43 Type 2 [1]
        - `'IEC1'`: IEC 61672-1 Class 1 [3]
        - `'IEC2'`: IEC 61672-1 Class 2 [3]
        By default, `'ANSI0'` (the strictest standard).
    must_satisfy_standard : bool
        If `fs_weighting` does not satisfy `standard`,
        raise a warning (if `must_satisfy_standard == False`)
        or an error (if `must_satisfy_standard == True`).

    Returns
    -------
    L_eq : float
        Equivalent continuous sound level.

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    [2] A. N. Rimell, N. J. Mansfield, and G. S. Paddan,
        “Design of digital filters for frequency weightings (A and C)
        required for risk assessments of workers exposed to noise,”
        Ind. Health, pp. 3–2013, 2014.
    [3] International Electrotechnical Commission, “Electroacoustics:
        Sound Level Meters. Specifications. Part 1,”
        International Electrotechnical Commission, 2013.
    """
    wave = weight_wave(wave, fs_orig,
                       weighting=weighting,
                       fs_weighting=fs_weighting,
                       standard=standard,
                       must_satisfy_standard=must_satisfy_standard)
    power_frames = 10*np.log10(
        np.var(frame_wave(wave, fs_weighting, frame_sec), axis=1)
    )

    if power_frames.shape[0] > 0:
        L_x = np.percentile(
            power_frames, 100 - np.array(x),
            interpolation='linear'
        )
    else:
        L_x = np.full(len(x), np.nan) if isinstance(x, Sized) else np.nan
    return L_x


def level_metrics(
    wave: np.ndarray,
    fs_orig: float,
    *,
    weighting: str = 'A',
    fs_weighting: float = 96_000.0,
    frame_sec: float = 1.0,
    standard: str = 'ANSI0',
    must_satisfy_standard: bool = True,
) -> Tuple[float, float, float, float]:
    """Calculate equivalent continuous sound level (L_eq)
    and percentile sound pressure level (L_5, L_50, L_95).

    The parameters for the weightings are referred to [1].
    The digital filter is designed by the bilinear transform [2].

    Parameters
    ----------
    wave : numpy.ndarray [shape=(wave_len, )]
        Audio wave.
    fs_orig : float
        Sampling frequency [Hz] of `wave`.
    weighting : str
        Frequency weighting of the sound pressure.
        `weighting` must be either `'A'`, `'B'`, `'C'`, or `'Z'`.
        By default `'A'`.
    fs_weighting : float
        Sampling frequency [Hz] for digital-filtering.
        By default `96_000.0`.
    frame_sec : float
        Frame width [second].
        By default, `1.0`.
        This parameter is used to calculate L_95, L_50, L_5.
    standard : str
        Standard referenced.
        If `fs_weighting` does not satisfy `standard`,
        raise a warning (if `must_satisfy_standard == False`)
        or an error (if `must_satisfy_standard == True`).
        Choose one of the following:
        - `'ANSI0'`: ANSI S.1-43 Type 0 [1]
        - `'ANSI1'`: ANSI S.1-43 Type 1 [1]
        - `'ANSI2'`: ANSI S.1-43 Type 2 [1]
        - `'IEC1'`: IEC 61672-1 Class 1 [3]
        - `'IEC2'`: IEC 61672-1 Class 2 [3]
        By default, `'ANSI0'` (the strictest standard).
    must_satisfy_standard : bool
        If `fs_weighting` does not satisfy `standard`,
        raise a warning (if `must_satisfy_standard == False`)
        or an error (if `must_satisfy_standard == True`).

    Returns
    -------
    L_eq : float
        Equivalent continuous sound level.
    L_5 : float
        Percentile sound pressure level (top 5%)
    L_50 : float
        Percentile sound pressure level (top 50%)
    L_95 : float
        Percentile sound pressure level (top 95%)

    References
    ----------
    [1] American National Standards Institute, “ANSI S1.43:
        Specifications for Integrating-averaging Sound Level Meters,”
        Acoustical Society of America, 1997.
    [2] A. N. Rimell, N. J. Mansfield, and G. S. Paddan,
        “Design of digital filters for frequency weightings (A and C)
        required for risk assessments of workers exposed to noise,”
        Ind. Health, pp. 3–2013, 2014.
    [3] International Electrotechnical Commission, “Electroacoustics:
        Sound Level Meters. Specifications. Part 1,”
        International Electrotechnical Commission, 2013.
    """
    wave = weight_wave(wave, fs_orig,
                       weighting=weighting,
                       fs_weighting=fs_weighting,
                       standard=standard,
                       must_satisfy_standard=must_satisfy_standard)
    L_eq = 10*np.log10(np.var(wave))
    power_frames = 10*np.log10(
        np.var(frame_wave(wave, fs_weighting, frame_sec), axis=1)
    )
    if power_frames.shape[0] > 0:
        L_5, L_50, L_95 = np.percentile(
            power_frames, [95.0, 50.0, 5.0],
            interpolation='linear'
        )
    else:
        L_5, L_50, L_95 = np.nan, np.nan, np.nan
    return L_eq, L_5, L_50, L_95
