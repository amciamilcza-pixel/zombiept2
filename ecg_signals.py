import os
import numpy as np
import wfdb         #PhysioNet library that reads MIT-BIH files (.dat and .hea)
                    #raw signal from .dat , sample rate and units from .hea

MITDB_DIR = "mitdb"     #MIT-BIH database with heartrates
NOISE_DIR = "nstdb"     #MIT-BIH database with noise
TARGET_FS = 500         #defines target sampling rate as 500 Hz

#loads a 5 seconds of ECG segment from the MIT-BIH database
#skips first 5 s because in real recording the electrodes are settling so the signal is often noisy 
def _load_record(record_id, duration=10.0, start_sec=5.0, channel=0):   
    #builds file path and reads it with wfdb 
    path = os.path.join(MITDB_DIR, record_id)       
    rec = wfdb.rdrecord(path)
    #converts integer from a file to a float
    fs_orig = rec.fs
    sig_all = rec.p_signal[:, channel].astype(float)   
    #converts seconds into sample indices
    start_sample = int(start_sec * fs_orig)             
    end_sample = start_sample + int(duration * fs_orig)
    sig = sig_all[start_sample:end_sample]

    #resamples the original frequency to target frequency of 500 Hz
    if fs_orig != TARGET_FS:
        n_out = int(len(sig) * TARGET_FS / fs_orig)
        t_in = np.linspace(0, 1, len(sig))
        t_out = np.linspace(0, 1, n_out)
        sig = np.interp(t_out, t_in, sig)

    #removes offset so signal is centered around zero
    sig = sig - np.mean(sig)

    #creates time values for the x-axis for plots
    t = np.arange(len(sig))/TARGET_FS
    return t, sig, TARGET_FS

#following three functions load records, respictively for healthy, zombie and infected cases 
def load_normal(duration=10.0):
    t, sig, _ = _load_record("100", duration=duration)
    return t, sig

def load_zombie(duration=10.0):
    t, sig, _ = _load_record("119", duration=duration)
    return t, -sig

def load_infected(duration=10.0):
    t, sig, _ = _load_record("203", duration=duration)
    return t, sig

#Loads a real noise signal from the MIT-BIH Noise Stress Test Database
def _load_noise_record(name, n_samples, fs, seed=0):
    #in the same way as _load_record:
    # builds file path, reads teh file, converts it to float, resamples to 500 Hz
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

    #picks n samples from a random position in the noise recording
    #same seed always gives the same chunk of the recording 
    rng = np.random.default_rng(seed)
    start = rng.integers(0, len(raw_rs) - n_samples)
    chunk = raw_rs[start:start + n_samples]

    #removes offset and scales noise to 1.0 amplitude
    chunk = chunk - np.mean(chunk)
    chunk = chunk/(np.sqrt(np.mean(chunk**2)) + 1e-10)

    return chunk

#adds real noises to the heartbeat
# bw - baseline wander
# em - electrode motion
# ma - muscle artefact
def add_real_noise(ecg, fs=500, snr_db=5, seed=99):
    N = len(ecg)
    t = np.arange(N) / fs
    sig_rms = np.sqrt(np.mean(ecg**2)) + 1e-10

    try:
        bw = _load_noise_record("bw", N, fs, seed=seed)
        em = _load_noise_record("em", N, fs, seed=seed + 1)
        ma = _load_noise_record("ma", N, fs, seed=seed + 2)
    except FileNotFoundError:
        print("Noise files not found. Using synthetic noise.")
        return add_noise(ecg, fs=fs, snr_db=snr_db, seed=seed)
    #combines the three noise types  
    mixed_noise = 0.45 * bw + 0.35 * em + 0.20 * ma
    #calculates signal-to-noise ratio
    noise_rms_target = sig_rms / (10 ** (snr_db/20))
    mixed_noise = mixed_noise / (np.sqrt(np.mean(mixed_noise**2)) + 1e-10)
    mixed_noise = mixed_noise * noise_rms_target

    hum = 0.25 * sig_rms * np.sin(2 * np.pi * 50 * t)
    return ecg + mixed_noise + hum

#adds synthetic fallback noise
def add_noise(ecg, fs=500, snr_db=5, seed=99):
    #in the same way as in add_real_noise:
    #generates a random number for a seed so same seed always gives the same noise
    rng = np.random.default_rng(seed)
    N = len(ecg)
    t = np.arange(N)/fs
    sig_rms = np.sqrt(np.mean(ecg**2)) + 1e-10


    noise_rms = sig_rms/(10 ** (snr_db / 20)) 
    white = rng.normal(0, noise_rms, N)         #white noise - pollutes whole spectrum uniformly
    baseline_wander = 0.30 * sig_rms * np.sin(2 * np.pi * 0.20 * t)     
    hum = 0.25 * sig_rms * np.sin(2 * np.pi * 50 * t)

    return ecg + white + baseline_wander + hum