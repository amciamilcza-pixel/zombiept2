"""
stress_tests.py
---------------
Structured robustness analysis for the Zombie Apocalypse ECG project.

Implements the three stress tests required by the rubric:
  1. Parameter sensitivity  — vary DFT window length N
  2. Noise robustness       — vary SNR, track heart-rate detection accuracy
  3. Failure / limitation   — spectral leakage + non-stationarity of arrhythmia

Each function returns data ready for plotting by run_me.py.
"""

import numpy as np
from ecg_signals import generate_normal, generate_arrhythmia, add_noise
from dft_filter import heart_rate_from_dft, spectral_magnitude, dft_bandpass


# ─────────────────────────────────────────────────────────────────────────────
# STRESS TEST 1 : Parameter Sensitivity — window length N
# ─────────────────────────────────────────────────────────────────────────────

def test_window_sensitivity(fs=500, duration=10.0):
    """
    Vary the DFT window length N and measure:
      - Frequency resolution  Δf = fs / N
      - Accuracy of heart-rate estimate (compare to ground truth 70 BPM)

    Returns dict with arrays for plotting.
    """
    t, ecg_clean = generate_normal(fs=fs, duration=duration)

    window_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    delta_f      = []
    hr_estimates = []
    spectra      = []   # (freqs, mag) per window

    for N in window_sizes:
        segment = ecg_clean[:N]
        freqs_half, mag_half, _ = spectral_magnitude(segment, fs)
        delta_f.append(fs / N)
        hr_estimates.append(heart_rate_from_dft(segment, fs))
        spectra.append((freqs_half, mag_half))

    return {
        "window_sizes": window_sizes,
        "delta_f":      delta_f,
        "hr_estimates": hr_estimates,
        "spectra":      spectra,
        "true_bpm":     70.0,
        "fs":           fs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRESS TEST 2 : Noise Robustness — vary SNR
# ─────────────────────────────────────────────────────────────────────────────

def test_noise_robustness(fs=500, duration=10.0, n_trials=8):
    """
    Add apocalypse noise at varying SNR levels to all three heart types.
    For each condition measure whether DFT can still recover the correct BPM.

    Returns dict with arrays for plotting.
    """
    from ecg_signals import generate_bradycardia

    snr_levels = np.linspace(-5, 30, 15)   # dB
    true_bpms  = {"normal": 70.0, "bradycardia": 30.0, "arrhythmia": 140.0}

    results = {}
    generators = {
        "normal":      generate_normal,
        "bradycardia": generate_bradycardia,
        "arrhythmia":  generate_arrhythmia,
    }

    for name, gen_fn in generators.items():
        t, ecg_clean = gen_fn(fs=fs, duration=duration)
        hr_at_snr = []
        hr_error  = []

        for snr in snr_levels:
            # Average over multiple noise seeds for stable estimates
            hrs = []
            for seed in range(n_trials):
                noisy = add_noise(ecg_clean, fs, snr_db=snr, seed=seed*17)
                # Apply bandpass before BPM estimation
                recovered, _, _ = dft_bandpass(noisy, fs, f_low=0.5, f_high=45.0)
                hrs.append(heart_rate_from_dft(recovered, fs))
            mean_hr = np.mean(hrs)
            hr_at_snr.append(mean_hr)
            hr_error.append(abs(mean_hr - true_bpms[name]))

        results[name] = {
            "snr_levels": snr_levels,
            "hr_at_snr":  hr_at_snr,
            "hr_error":   hr_error,
            "true_bpm":   true_bpms[name],
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STRESS TEST 3 : Failure / Limitation — non-stationarity & spectral leakage
# ─────────────────────────────────────────────────────────────────────────────

def test_failure_cases(fs=500):
    """
    Demonstrates two fundamental DFT failure modes:

    A) Spectral leakage
       - Take a short segment of the normal ECG (non-integer number of periods)
       - Compare rectangular window vs. Hann window spectrum
       - Show how leakage smears the R-peak frequency

    B) Non-stationarity of arrhythmia
       - The global DFT of the arrhythmia signal gives a *misleading* average
         that looks like a single frequency — hiding the beat-to-beat variability
       - Compare global DFT vs. short-time (sliding window) DFT to show what
         the global DFT misses

    Returns dict with all data needed for plotting.
    """
    # ── A: Spectral leakage ────────────────────────────────────────────────
    t_full, ecg_normal = generate_normal(fs=fs, duration=10.0)

    # Deliberately awkward window — not aligned to heartbeat period
    N_leak = 300   # ~0.6 s, not a multiple of 70 BPM period (~0.857 s)
    seg = ecg_normal[:N_leak]
    t_seg = t_full[:N_leak]

    freqs_leak = np.fft.rfftfreq(N_leak, d=1.0/fs)
    mag_rect   = np.abs(np.fft.rfft(seg))
    mag_hann   = np.abs(np.fft.rfft(seg * np.hanning(N_leak)))

    # ── B: Non-stationarity — global vs sliding DFT ────────────────────────
    t_arr, ecg_arr = generate_arrhythmia(fs=fs, duration=10.0)

    # Global DFT
    freqs_global = np.fft.rfftfreq(len(ecg_arr), d=1.0/fs)
    mag_global   = np.abs(np.fft.rfft(ecg_arr))

    # Sliding-window DFT (manual STFT — using only numpy, no scipy)
    win_len   = 256          # ~0.5 s window
    hop       = 64           # step size in samples
    n_frames  = (len(ecg_arr) - win_len) // hop
    stft_mag  = np.zeros((win_len // 2 + 1, n_frames))
    stft_times = np.zeros(n_frames)

    hann_win = np.hanning(win_len)
    for i in range(n_frames):
        start = i * hop
        frame = ecg_arr[start : start + win_len] * hann_win
        stft_mag[:, i] = np.abs(np.fft.rfft(frame))
        stft_times[i]  = (start + win_len // 2) / fs

    stft_freqs = np.fft.rfftfreq(win_len, d=1.0/fs)

    return {
        # Leakage data
        "leakage": {
            "t_seg":      t_seg,
            "seg":        seg,
            "freqs":      freqs_leak,
            "mag_rect":   mag_rect,
            "mag_hann":   mag_hann,
            "N":          N_leak,
            "fs":         fs,
        },
        # Non-stationarity data
        "nonstationarity": {
            "t":           t_arr,
            "ecg":         ecg_arr,
            "freqs_global": freqs_global,
            "mag_global":   mag_global,
            "stft_mag":     stft_mag,
            "stft_times":   stft_times,
            "stft_freqs":   stft_freqs,
        },
    }
