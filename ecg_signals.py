import os
import numpy as np
import wfdb

MITDB_DIR = "mitdb"
NOISE_DIR = "nstdb"
TARGET_FS = 500


def _load_record(record_id, duration=10.0, start_sec=5.0, channel=0):
    """
    Load a short ECG segment from the MIT-BIH database.
    The signal is resampled to 500 Hz and centered around zero.
    """
    path = os.path.join(MITDB_DIR, record_id)
    rec = wfdb.rdrecord(path)

    fs_orig = rec.fs
    sig_all = rec.p_signal[:, channel].astype(float)

    start_samp = int(start_sec * fs_orig)
    end_samp = start_samp + int(duration * fs_orig)
    sig = sig_all[start_samp:end_samp]

    if fs_orig != TARGET_FS:
        n_out = int(len(sig) * TARGET_FS / fs_orig)
        t_in = np.linspace(0, 1, len(sig))
        t_out = np.linspace(0, 1, n_out)
        sig = np.interp(t_out, t_in, sig)

    sig = sig - np.mean(sig)

    t = np.arange(len(sig)) / TARGET_FS
    return t, sig, TARGET_FS


def generate_normal(duration=10.0):
    """
    Record 100: normal ECG.
    """
    t, sig, _ = _load_record("100", duration=duration)
    return t, sig


def generate_bradycardia(duration=10.0):
    """
    Record 119: bradycardia ECG.
    It is inverted and reduced in amplitude for the zombie effect.
    """
    t, sig, _ = _load_record("119", duration=duration)
    return t, -sig


def generate_arrhythmia(duration=10.0):
    """
    Record 203: irregular arrhythmia ECG.
    Used as the infected / turning heart.
    """
    t, sig, _ = _load_record("203", duration=duration)
    return t, sig


def _load_noise_record(name, n_samples, fs, seed=0):
    """
    Load a real noise signal from the MIT-BIH Noise Stress Test Database.
    """
    path = os.path.join(NOISE_DIR, name)
    rec = wfdb.rdrecord(path)

    raw = rec.p_signal[:, 0].astype(float)
    fs_orig = rec.fs

    n_resampled = int(len(raw) * fs / fs_orig)
    raw_rs = np.interp(
        np.linspace(0, 1, n_resampled),
        np.linspace(0, 1, len(raw)),
        raw
    )

    rng = np.random.default_rng(seed)
    start = rng.integers(0, len(raw_rs) - n_samples)
    chunk = raw_rs[start:start + n_samples]

    chunk = chunk - np.mean(chunk)
    chunk = chunk / (np.sqrt(np.mean(chunk**2)) + 1e-10)

    return chunk


def add_real_noise(ecg, fs=500, snr_db=5, seed=99):
    """
    Add real NSTDB noise to the ECG.
    Also adds synthetic 50 Hz hum for the notch filter demonstration.
    """
    N = len(ecg)
    t = np.arange(N) / fs
    sig_rms = np.sqrt(np.mean(ecg**2)) + 1e-10

    try:
        bw = _load_noise_record("bw", N, fs, seed=seed)
        em = _load_noise_record("em", N, fs, seed=seed + 1)
        ma = _load_noise_record("ma", N, fs, seed=seed + 2)
    except FileNotFoundError:
        print("Real noise files not found. Using synthetic noise instead.")
        return add_noise(ecg, fs=fs, snr_db=snr_db, seed=seed)

    mixed_noise = 0.45 * bw + 0.35 * em + 0.20 * ma

    noise_rms_target = sig_rms / (10 ** (snr_db / 20))
    mixed_noise = mixed_noise / (np.sqrt(np.mean(mixed_noise**2)) + 1e-10)
    mixed_noise = mixed_noise * noise_rms_target

    hum = 0.25 * sig_rms * np.sin(2 * np.pi * 50 * t)

    return ecg + mixed_noise + hum


def add_noise(ecg, fs=500, snr_db=5, seed=99):
    """
    Add synthetic fallback noise:
    white noise, baseline wander, and 50 Hz hum.
    """
    rng = np.random.default_rng(seed)
    N = len(ecg)
    t = np.arange(N) / fs
    sig_rms = np.sqrt(np.mean(ecg**2)) + 1e-10

    noise_rms = sig_rms / (10 ** (snr_db / 20))
    white = rng.normal(0, noise_rms, N)
    wander = 0.30 * sig_rms * np.sin(2 * np.pi * 0.20 * t)
    hum = 0.25 * sig_rms * np.sin(2 * np.pi * 50 * t)

    return ecg + white + wander + hum