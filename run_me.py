"""
run_me.py
Main script for the Zombie ECG DFT project.
Generates the final four figures.
"""

import os

from ecg_signals import (
    generate_normal,
    generate_arrhythmia,
    generate_bradycardia,
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


print("Loading ECG signals...")

t, ecg_normal = generate_normal(duration=DURATION)
_, ecg_arrhythmia = generate_arrhythmia(duration=DURATION)
_, ecg_bradycardia = generate_bradycardia(duration=DURATION)

signals_clean = {
    "normal": ecg_normal,
    "arrhythmia": ecg_arrhythmia,
    "bradycardia": ecg_bradycardia,
}

print("Clean signal BPM estimates:")
for name, signal in signals_clean.items():
    bpm = heart_rate_dft(signal, FS)
    print(f"{name}: {bpm:.1f} BPM")


print("\nAdding noise and recovering signals...")

signals_noisy = {}
signals_recovered = {}
recovery_info = {}

for name, clean_signal in signals_clean.items():
    noisy = add_real_noise(clean_signal, fs=FS, snr_db=SNR_DB)
    recovered, X_noisy, X_recovered, freqs = recover_ecg(noisy, FS)

    signals_noisy[name] = noisy
    signals_recovered[name] = recovered

    recovery_info[name] = {
        "X_noisy": X_noisy,
        "X_recovered": X_recovered,
        "freqs": freqs,
    }

    print(
        f"{name}: "
        f"clean = {heart_rate_dft(clean_signal, FS):.1f} BPM, "
        f"noisy = {heart_rate_dft(noisy, FS):.1f} BPM, "
        f"recovered = {heart_rate_dft(recovered, FS):.1f} BPM"
    )


print("\nGenerating figures...")

plot_main_noisy_vs_recovered(t, signals_noisy, signals_recovered, FS)

plot_dft_filter_explanation(
    recovery_info["normal"]["freqs"],
    recovery_info["normal"]["X_noisy"],
    recovery_info["normal"]["X_recovered"],
    example_label="normal",
)

window_data = window_sensitivity(fs=FS, duration=DURATION)
noise_data = noise_robustness(fs=FS, duration=DURATION)
plot_stress_tests_summary(window_data, noise_data)

failure_data = failure_cases(fs=FS)
plot_dft_failure_case(failure_data)


expected_figures = [
    "fig1_main_noisy_vs_recovered.png",
    "fig2_dft_filter_explanation.png",
    "fig3_stress_tests_summary.png",
    "fig4_dft_failure_case.png",
]

print("\nDone. Checking final figures:")

for fig_name in expected_figures:
    fig_path = os.path.join(FIG_DIR, fig_name)

    if os.path.exists(fig_path):
        print(f"✓ {fig_name} created")
    else:
        print(f"✗ {fig_name} missing")