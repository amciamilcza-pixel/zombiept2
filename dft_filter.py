import numpy as np


def spectral_magnitude(signal, fs):
    """
    Compute the one-sided DFT magnitude spectrum.
    """
    N = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)

    half = N // 2
    return freqs[:half], np.abs(X[:half]), X


def dft_bandpass(signal, fs, f_low=0.5, f_high=45.0):
    """
    Keep only frequencies between f_low and f_high using the DFT.
    """
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/fs)

    mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
    X_filtered = X * mask

    filtered = np.fft.ifft(X_filtered).real
    return filtered, X_filtered, freqs


def recover_ecg(noisy_signal, fs, f_low=0.5, f_high=45.0):
    """
    Recover ECG by keeping only the main ECG frequency band using the DFT.
    Frequencies below f_low and above f_high are removed.
    """
    recovered, X_recovered, freqs = dft_bandpass(noisy_signal, fs, f_low, f_high)

    X_noisy = np.fft.fft(noisy_signal)

    return recovered, X_noisy, X_recovered, freqs


def heart_rate_dft(signal, fs, search_band=(0.5, 3.5)):
    """
    Estimate BPM from the strongest DFT peak in a realistic heart-rate band.
    """
    freqs, mag, _ = spectral_magnitude(signal, fs)

    mask = (freqs >= search_band[0]) & (freqs <= search_band[1])
    f_band = freqs[mask]
    mag_band = mag[mask]

    if len(f_band) == 0:
        return 0.0

    dominant_freq = f_band[np.argmax(mag_band)]
    bpm = dominant_freq * 60

    return bpm