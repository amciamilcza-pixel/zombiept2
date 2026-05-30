import numpy as np

# Compute DFT - which frequencies are present in the signal and shows their amplitude
def spectral_magnitude(signal, fs):
    # Converts ECG signal in frequency domain (X contains complex values - how much of each frequency is present
    N = len(signal)
    X = np.fft.fft(signal)        
    freqs = np.fft.fftfreq(N, d=1/fs)       # Creates matching frequency values

    half = N//2
    return freqs[:half], np.abs(X[:half]), X # Return only the positive frequencies for plotting, and the full DFT

#Keep only frequencies within the useful ECG range (between f_low and f_high)
def dft_bandpass(signal, fs, f_low=0.5, f_high=45):
    X = np.fft.fft(signal) 
    freqs = np.fft.fftfreq(len(signal), d=1/fs) 

    mask = (np.abs(freqs)>=f_low) & (np.abs(freqs)<=f_high)     # Filter that checks if the given frequency is within the useful range
    X_filtered = X*mask             # Applies bandpass filter, automatically removes the 50Hz powerline signal

    filtered = np.fft.ifft(X_filtered).real # Convert signal back to time domain
    return filtered, X_filtered, freqs

# Recover the previously kept frequencies
def recover_ecg(noisy_signal, fs, f_low=0.5, f_high=45):
    recovered, X_recovered, freqs = dft_bandpass(noisy_signal, fs, f_low, f_high) # Apply bandpass filter
    X_noisy = np.fft.fft(noisy_signal) # Compute the DFT of the noisy signal before filtering

    return recovered, X_noisy, X_recovered, freqs

# Estimate BPM using the strongest frequency using DFT spectrum, heartrate is between 0.5 and 3.5 Hz
def heart_rate_dft(signal, fs, search_band=(0.5, 3.5)):
    freqs, mag, _ = spectral_magnitude(signal, fs) # Compute DFT spectrum

    mask = (freqs >= search_band[0]) & (freqs <= search_band[1]) # Keep only useful range (0.5-3.5Hz)
    f_band = freqs[mask]        # Store the useful frequencies
    mag_band = mag[mask]        # Store the useful magnitudes

    # Safety check: if there are no frequencies in the selected range, returns 0 BPM (otherwise crashes)
    if len(f_band) == 0:
        return 0

    # Find the strongest frequency in the band and calculate BPM
    dominant_freq = f_band[np.argmax(mag_band)] 
    bpm = dominant_freq * 60

    return bpm