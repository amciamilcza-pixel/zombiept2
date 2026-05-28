"""
dft_filter.py
-------------
DFT-based signal recovery for the Zombie Apocalypse ECG project.

All filtering is done EXPLICITLY in the frequency domain using numpy FFT.
The DFT is the core analytical tool — not just a visualisation step.

Key insight: the DFT decomposes the signal into frequency components.
By zeroing bins outside the cardiac band and inverting, we recover only
the heartbeat — everything else (noise, hum, wander) is discarded.

Functions
---------
dft_bandpass        : zero DFT bins outside [f_low, f_high], then IDFT
dft_notch           : zero DFT bins around a specific frequency
recover_ecg         : full pipeline: notch + bandpass + inverse DFT
spectral_magnitude  : one-sided magnitude spectrum
heart_rate_from_dft : find fundamental frequency peak → BPM
"""

import numpy as np
from scipy.signal import find_peaks


def spectral_magnitude(signal, fs):
    """
    One-sided DFT magnitude spectrum.

    Returns
    -------
    freqs : frequency axis Hz (positive only)
    mag   : magnitude spectrum
    X     : full complex DFT (needed for inverse transform)
    """
    N = len(signal)
    X     = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    half  = N // 2
    return freqs[:half], np.abs(X[:half]), X


def dft_bandpass(signal, fs, f_low, f_high):
    """
    Ideal rectangular bandpass filter in the DFT domain.

    Steps
    -----
    1. Compute DFT  X[k] = sum_n x[n] * e^{-j2pi*k*n/N}
    2. Zero all bins k where |f[k]| < f_low  or  |f[k]| > f_high
    3. Inverse DFT  x_filt[n] = (1/N) * sum_k X_filt[k] * e^{j2pi*k*n/N}

    This is the fundamental DFT filtering operation — we operate
    directly on the spectrum, not with a convolution kernel.
    """
    N      = len(signal)
    X      = np.fft.fft(signal)
    freqs  = np.fft.fftfreq(N, d=1.0 / fs)

    # Passband mask (both positive and negative frequencies)
    mask       = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
    X_filtered = X * mask

    filtered = np.fft.ifft(X_filtered).real
    return filtered, X_filtered, freqs


def dft_notch(signal, fs, f_notch, bandwidth=2.0):
    """
    Notch filter in DFT domain: zero bins within ±bandwidth/2 of f_notch.
    Used to remove 50 Hz power-line hum before bandpass filtering.
    """
    N      = len(signal)
    X      = np.fft.fft(signal)
    freqs  = np.fft.fftfreq(N, d=1.0 / fs)

    notch_mask    = np.abs(np.abs(freqs) - f_notch) < bandwidth / 2
    X[notch_mask] = 0

    return np.fft.ifft(X).real


def recover_ecg(noisy_signal, fs, f_low=0.5, f_high=45.0, notch_hz=50.0):
    """
    Full DFT-based ECG recovery pipeline:
      Step 1 — Notch filter  : remove 50 Hz hum
      Step 2 — Bandpass      : keep only 0.5–45 Hz (cardiac content)
      Step 3 — Inverse DFT   : back to time domain

    Returns recovered signal + DFT arrays for before/after comparison plots.
    """
    after_notch              = dft_notch(noisy_signal, fs, notch_hz)
    recovered, X_rec, freqs  = dft_bandpass(after_notch, fs, f_low, f_high)
    X_noisy                  = np.fft.fft(noisy_signal)
    return recovered, X_noisy, X_rec, freqs


def heart_rate_from_dft(signal, fs, search_band=(0.4, 3.5)):
    """
    Estimate heart rate (BPM) from the DFT by finding the FUNDAMENTAL
    frequency — the lowest significant spectral peak in the cardiac band.

    Why lowest peak, not highest?
    The ECGSYN model (used by NeuroKit2) generates strong harmonics.
    The 2nd harmonic (2× fundamental) can have higher magnitude than the
    fundamental itself. We want the FUNDAMENTAL (f0) because:
        f0 [Hz] × 60 = heart rate [BPM]

    Strategy: find all peaks above 30% of band maximum, return lowest.
    This correctly identifies f0 even when harmonics are stronger.

    Returns BPM (float), or 0 if no peak found.
    """
    N       = len(signal)
    X       = np.fft.fft(signal)
    freqs   = np.fft.fftfreq(N, d=1.0 / fs)
    half    = N // 2
    f_pos   = freqs[:half]
    mag_pos = np.abs(X[:half])

    # Restrict to search band
    mask    = (f_pos >= search_band[0]) & (f_pos <= search_band[1])
    if not np.any(mask):
        return 0.0

    f_band = f_pos[mask]
    m_band = mag_pos[mask]

    # Minimum spacing between peaks: 0.2 Hz (avoids noise spikes)
    min_dist = max(1, int(0.2 / (f_pos[1] - f_pos[0])))
    threshold = 0.30 * m_band.max()

    peaks, _ = find_peaks(m_band, height=threshold, distance=min_dist)
    if len(peaks) == 0:
        # Fallback: raw argmax
        return f_band[np.argmax(m_band)] * 60.0

    # Return lowest-frequency significant peak = fundamental
    return f_band[peaks[0]] * 60.0
