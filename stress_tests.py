import numpy as np
from ecg_signals import load_normal, load_infected, load_zombie, add_noise
from dft_filter import heart_rate_dft, spectral_magnitude, dft_bandpass

## PARAMETER SENSITIVITY - observation window length

def window_sensitivity(fs=500, duration=10.0):
    #generates clean 70 BPM ECG
    t, ecg_clean = load_normal(duration=duration) #Hz, seconds

    #test different window sizes samples
    window_sizes = [64, 128, 256, 512, 1024, 2048, 4096] 
    delta_f = []
    hr_estimates = []
    spectra = []

    for N in window_sizes:
        segment = ecg_clean[:N]         #takes only first N samples from the ECG
        freqs_half, mag_half, _ = spectral_magnitude(segment, fs)   #Hz, gets frequency spectrum on these samples
        delta_f.append(fs/N)
        hr_estimates.append(heart_rate_dft(segment, fs))       #estimates heartrate
        spectra.append((freqs_half, mag_half))

    return {
        "window_sizes": window_sizes,
        "delta_f": delta_f,             #Hz   
        "hr_estimates": hr_estimates,   #BPM
        "spectra": spectra,             #list of (Hz, amplitude) tuples
        "true_bpm": 70.0,               #BPM
        "fs": fs,                       #Hz
    }

# NOISE ROBUSTNESS

#runs 8 trials with different random noise seeds
def noise_robustness(fs=500, duration=10.0, n_trials=8):

    snr_levels = np.linspace(-5, 30, 15)   #dB
    true_bpms = {"normal": 70.0, "arrhythmia": 140.0, "bradycardia": 30.0}

    results = {}
    generators = {
        "normal": load_normal,
        "arrhythmia": load_infected,
        "bradycardia": load_zombie,
    }

    for name, gen_fn in generators.items():
        t, ecg_clean = gen_fn(duration=duration)     #Hz, seconds
        hr_snr = []      #BPM, estimated heart rate at each SNR
        hr_error = []       #BPM, errror at each SNR

        for snr in snr_levels:
            #takes an average over multiple noise seeds for stable estimates
            hrs = []
            for seed in range(n_trials):
                noisy = add_noise(ecg_clean, fs, snr_db=snr, seed=seed*17)      #Hz, dB
                #applies bandpass and estimates BPM 
                recovered, _, _ = dft_bandpass(noisy, fs, f_low=0.5, f_high=45.0)   #Hz, Hz, Hz
                hrs.append(heart_rate_dft(recovered, fs))      #BPM, Hz

            mean_hr = np.mean(hrs)
            hr_snr.append(mean_hr)
            hr_error.append(abs(mean_hr - true_bpms[name]))

        results[name] = {
            "snr_levels": snr_levels,       #dB
            "hr_snr": hr_snr,         #BPM    
            "hr_error": hr_error,           #BPM       
            "true_bpm": true_bpms[name],    #BPM
        }

    return results

## FAILURE CASE

def failure_cases(fs=500):
    t_arr, ecg_arr = load_infected(duration=10.0)      #Hz, seconds

    #global DFT
    freqs_global = np.fft.rfftfreq(len(ecg_arr), d=1.0/fs)          #Hz
    mag_global = np.abs(np.fft.rfft(ecg_arr))

    ##STFT (Short-Time Fourier Transform)
    win_len = 256       #sets window length to 256
    hop = 64            #step size between consecutive windows
    n_frames = (len(ecg_arr) - win_len) // hop

    stft_mag = np.zeros((win_len // 2 + 1, n_frames))     #amplitude, shape (freq_bins, frames)
    stft_times = np.zeros(n_frames)             #seconds (centre time of each frame)
    hann_win = np.hanning(win_len)

    for i in range(n_frames):
        start = i*hop                            
        frame = ecg_arr[start:start + win_len]*hann_win
        stft_mag[:, i] = np.abs(np.fft.rfft(frame))     #amplitude
        stft_times[i] = (start + win_len//2)/fs    #seconds

    stft_freqs = np.fft.rfftfreq(win_len, d=1.0/fs)   #Hz

    return {
        "nonstationarity": {
            "t": t_arr,                      #seconds
            "ecg": ecg_arr,                  #amplitude 
            "freqs_global": freqs_global,    #Hz
            "mag_global": mag_global,        #amplitude
            "stft_mag": stft_mag,            #amplitude, axes: (freq_bins × frames)
            "stft_times": stft_times,        #seconds
            "stft_freqs": stft_freqs,        #Hz
        },
    }