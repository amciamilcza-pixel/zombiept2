import numpy as np

from ecg_signals import (
    generate_normal,
    generate_arrhythmia,
    generate_bradycardia,
    add_noise,
)

from dft_filter import heart_rate_dft, spectral_magnitude, dft_bandpass


def window_sensitivity(fs=500, duration=10.0):
    """
    Test how DFT window length affects BPM estimation.
    Short windows have poor frequency resolution.
    """
    t, ecg_clean = generate_normal(duration=duration)

    window_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    delta_f = []
    hr_estimates = []
    spectra = []

    for N in window_sizes:
        segment = ecg_clean[:N]

        freqs_half, mag_half, _ = spectral_magnitude(segment, fs)

        delta_f.append(fs / N)
        hr_estimates.append(heart_rate_dft(segment, fs))
        spectra.append((freqs_half, mag_half))

    return {
        "window_sizes": window_sizes,
        "delta_f": delta_f,
        "hr_estimates": hr_estimates,
        "spectra": spectra,
        "true_bpm": 70.0,
        "fs": fs,
    }


def noise_robustness(fs=500, duration=10.0, n_trials=5):
    """
    Test how added synthetic noise affects BPM estimation.
    Synthetic noise is used here so the SNR can be controlled.
    """
    snr_levels = np.linspace(-5, 30, 15)

    true_bpms = {
        "normal": 70.0,
        "bradycardia": 30.0,
        "arrhythmia": 140.0,
    }

    generators = {
        "normal": generate_normal,
        "bradycardia": generate_bradycardia,
        "arrhythmia": generate_arrhythmia,
    }

    results = {}

    for name, gen_fn in generators.items():
        t, ecg_clean = gen_fn(duration=duration)

        hr_snr = []
        hr_error = []

        for snr in snr_levels:
            bpm_trials = []

            for seed in range(n_trials):
                noisy = add_noise(ecg_clean, fs, snr_db=snr, seed=seed)
                recovered, _, _ = dft_bandpass(noisy, fs, f_low=0.5, f_high=45.0)
                bpm_trials.append(heart_rate_dft(recovered, fs))

            mean_bpm = np.mean(bpm_trials)
            hr_snr.append(mean_bpm)
            hr_error.append(abs(mean_bpm - true_bpms[name]))

        results[name] = {
            "snr_levels": snr_levels,
            "hr_snr": hr_snr,
            "hr_error": hr_error,
            "true_bpm": true_bpms[name],
        }

    return results


def failure_cases(fs=500):
    """
    Show why one global DFT can be misleading for an irregular signal.
    We compare one DFT of the whole signal to repeated DFTs over short windows.
    """
    t_arr, ecg_arr = generate_arrhythmia(duration=10.0)

    freqs_global = np.fft.rfftfreq(len(ecg_arr), d=1/fs)
    mag_global = np.abs(np.fft.rfft(ecg_arr))

    win_len = 256
    hop = 64

    n_frames = (len(ecg_arr) - win_len) // hop

    stft_mag = np.zeros((win_len // 2 + 1, n_frames))
    stft_times = np.zeros(n_frames)
    window = np.hanning(win_len)

    for i in range(n_frames):
        start = i * hop
        frame = ecg_arr[start:start + win_len] * window

        stft_mag[:, i] = np.abs(np.fft.rfft(frame))
        stft_times[i] = (start + win_len / 2) / fs

    stft_freqs = np.fft.rfftfreq(win_len, d=1/fs)

    return {
        "nonstationarity": {
            "t": t_arr,
            "ecg": ecg_arr,
            "freqs_global": freqs_global,
            "mag_global": mag_global,
            "stft_mag": stft_mag,
            "stft_times": stft_times,
            "stft_freqs": stft_freqs,
        }
    }