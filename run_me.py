"""
run_me.py
Main script for the Zombie ECG DFT project
Generates the final four figures
"""

import os

from ecg_signals import (
    load_normal,
    load_infected,
    load_zombie,
    add_real_noise,
)

from dft_filter import recover_ecg, heart_rate_dft

from stress_tests import (
    window_sensitivity,
    noise_robustness,
    failure_cases,
)

from plots import (
    plot_main_noisy_vs_recovered,
    plot_dft_filter_explanation,
    plot_stress_tests_summary,
    plot_dft_failure_case,
)


FS = 500
DURATION = 10.0
SNR_DB = 5

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Remove old output files so we do not accidentally use outdated figures
for filename in os.listdir(FIG_DIR):
    if filename.endswith(".png"):
        os.remove(os.path.join(FIG_DIR, filename))

print("Old figure files removed.")


# Load the three ECG signals - normal, early-stage(arrythmia) and zombie(bradycardia)
print("Loading ECG signals...")

t, ecg_normal = load_normal(duration=DURATION)
_, ecg_arrhythmia = load_infected(duration=DURATION)
_, ecg_bradycardia = load_zombie(duration=DURATION)

# Create a dictionary with labels for each signal
signals_clean = {
    "normal": ecg_normal,
    "arrhythmia": ecg_arrhythmia,
    "bradycardia": ecg_bradycardia,
}

# Estimate the heartbeat BPM using DFT
print("Clean signal BPM estimates:")
for name, signal in signals_clean.items():
    bpm = heart_rate_dft(signal, FS)
    print(f"{name}: {bpm:.1f} BPM")


print("\nAdding noise and recovering signals...")

signals_noisy = {}
signals_recovered = {}
recovery_info = {}

# Add noise to each signal, recover it with DFT bandpass and store results
# Returns cleaned ECG signal in time domain, DFT spectrum of noisy signal, DFT spectrum after filtering and frequency axis in Hz
for name, clean_signal in signals_clean.items():
    noisy = add_real_noise(clean_signal, fs=FS, snr_db=SNR_DB)
    recovered, X_noisy, X_recovered, freqs = recover_ecg(noisy, FS)

    # Stores noisy and recovered signals for each heartbeat type in dictionaries
    signals_noisy[name] = noisy
    signals_recovered[name] = recovered

    # Saves noisy and recovered DFT spectrums for figure 2, as well as the frequency axis
    recovery_info[name] = {
        "X_noisy": X_noisy,
        "X_recovered": X_recovered,
        "freqs": freqs,
    }

    # Print BPM before noise, after noise and after filtering to see if DFT improved the reading
    print(
        f"{name}: "
        f"clean = {heart_rate_dft(clean_signal, FS):.1f} BPM, "
        f"noisy = {heart_rate_dft(noisy, FS):.1f} BPM, "
        f"recovered = {heart_rate_dft(recovered, FS):.1f} BPM"
    )


print("\nGenerating figures...")

# Create figure 1 to show noisy vs recovered ECG for each case
plot_main_noisy_vs_recovered(t, signals_noisy, signals_recovered, FS)

# Create figure 2 to visualize the DFT spectrum before and after filtering using the normal heartbeat
plot_dft_filter_explanation(
    recovery_info["normal"]["freqs"],
    recovery_info["normal"]["X_noisy"],
    recovery_info["normal"]["X_recovered"],
    example_label="normal",
)

# Create figure 3 to visualise the stress tests
window_data = window_sensitivity(fs=FS, duration=DURATION) # checks how window size impacts BPM estimate
noise_data = noise_robustness(fs=FS, duration=DURATION) # checks how BPM estimate works under different noise levels
plot_stress_tests_summary(window_data, noise_data)

# Create figure 4 to run the failure test: runs arrythmia signal and compares one DFT over whole signal vs repeated DFTs over short windows
failure_data = failure_cases(fs=FS)
plot_dft_failure_case(failure_data)


expected_figures = [
    "fig1_main_noisy_vs_recovered.png",
    "fig2_dft_filter_explanation.png",
    "fig3_stress_tests_summary.png",
    "fig4_dft_failure_case.png",
]

print("\nDone. Checking final figures:")

# Creates full path to each figure
for fig_name in expected_figures:
    fig_path = os.path.join(FIG_DIR, fig_name)

    if os.path.exists(fig_path):
        print(f"YES, {fig_name} created")
    else:
        print(f"NO, {fig_name} missing")