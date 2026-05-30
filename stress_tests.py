import numpy as np

from ecg_signals import (load_normal, load_infected, load_zombie, add_noise,)
from dft_filter import heart_rate_dft, spectral_magnitude, dft_bandpass

## PARAMETER SENSITIVITY - observation window length

def window_sensitivity(fs=500, duration=10.0):
   #generates clean 75 BPM ECG
    t, ecg_clean = load_normal(fs=fs, duration=duration)   #Hz, seconds

    #test different window sizes samples
    window_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    delta_f = []
    hr_estimates = []
    spectra = []

    for N in window_sizes:
        segment = ecg_clean[:N]         #takes only first N samples from the ECG
        freqs_half, mag_half, _ = spectral_magnitude(segment, fs)   #Hz, gets frequency spectrum on these samples
        delta_f.append(fs / N)
        hr_estimates.append(heart_rate_dft(segment, fs))      #estimates heartrate
        spectra.append((freqs_half, mag_half))

    return {
        "window_sizes":window_sizes,
        "delta_f":delta_f,             #Hz   
        "hr_estimates":hr_estimates,   #BPM
        "spectra":spectra,             #list of (Hz, amplitude) tuples
        "true_bpm":70.0,               #BPM
        "fs":fs,                       #Hz
    }

# NOISE ROBUSTNESS

#tests how added noise affects heartrate
#runs 5 trials with different random noise
def noise_robustness(fs=500, duration=10.0, n_trials=5):
    snr_levels = np.linspace(-5, 30, 15)         #dB
    true_bpms = {
        "normal": 75.0,         #mean BPM of record 100
        "zombie": 48.0,         #mean BPM of record 119
        "infected": 88.0,       #mean BPM of record 203
    }
    generators = {
        "normal":load_normal,
        "zombie":load_zombie,
        "infected":load_infected,
    }
    results = {}

    for name, gen_fn in generators.items():
        t, ecg_clean = gen_fn(fs=fs, duration=duration)     #Hz, seconds
        hr_snr = []      #BPM, estimated heart rate at each SNR
        hr_error = []       #BPM, errror at each SNR

        for snr in snr_levels:
            bpm_trials = []
            #takes an average over multiple noise seeds for stable estimates
            for seed in range(n_trials):
                noisy = add_noise(ecg_clean, fs, snr_db=snr, seed=seed)      #Hz, dB
                #applies bandpass and estimates BPM
                recovered, _, _ = dft_bandpass(noisy, fs, f_low=0.5, f_high=45.0)   #Hz, Hz, Hz
                bpm_trials.append(heart_rate_dft(recovered, fs))      #BPM, Hz
            mean_bpm = np.mean(bpm_trials)
            hr_snr.append(mean_bpm)
            hr_error.append(abs(mean_bpm - true_bpms[name]))

        results[name] = {
            "snr_levels": snr_levels,           #dB
            "hr_snr": hr_snr,                   #BPM
            "hr_error": hr_error,               #BPM
            "true_bpm": true_bpms[name],        #BPM
        }
    return results

## FAILURE CASE - irregular signals

#compares one DFT of the whole signal to repeated DFTs over short windows 
def failure_cases(fs=500):
    t_arr, ecg_arr = load_infected(duration=10.0)      #Hz, seconds
    #DFT of the whole signal
    #rfft of a real signal of length N returns N//2+1 unique frequency bins
    stft_mag = np.zeros((win_len // 2 + 1, n_frames))
    freqs_global = np.fft.rfftfreq(len(ecg_arr), d=1/fs)     #Hz
    mag_global = np.abs(np.fft.rfft(ecg_arr))

    #DFTs over short windows (Short-Time Fourier Transform STFT)
    win_len = 256       #sets window length to 256
    hop = 64            #step size between consecutive windows
    n_frames = (len(ecg_arr) - win_len) // hop
    stft_mag = np.zeros((win_len // 2 + 1, n_frames))     #amplitude, shape (freq_bins, frames)
    stft_times = np.zeros(n_frames)            #seconds (centre time of each frame)
    window = np.hanning(win_len)

    for i in range(n_frames):
        start = i * hop
        frame = ecg_arr[start:start + win_len] * window
        stft_mag[:, i] = np.abs(np.fft.rfft(frame))         #amplitude
        stft_times[i] = (start + win_len / 2) / fs          #seconds

    stft_freqs = np.fft.rfftfreq(win_len, d=1/fs)           #Hz

    return {
        "nonstationarity": {
            "t":t_arr,                      #seconds
            "ecg":ecg_arr,                  #amplitude 
            "freqs_global":freqs_global,    #Hz
            "mag_global":mag_global,        #amplitude
            "stft_mag":stft_mag,            #amplitude, axes: (freq_bins × frames)
            "stft_times":stft_times,        #seconds
            "stft_freqs":stft_freqs,        #Hz
        },
    }