"""
run_me.py
=========
Zombie Apocalypse ECG — DFT Signals & Systems Project

This version generates only the four video-ready figures:
  1. fig1_main_noisy_vs_recovered.png
  2. fig2_dft_filter_explanation.png
  3. fig3_stress_tests_summary.png
  4. fig4_dft_failure_case.png

Story/order everywhere:
  normal → arrhythmia / turning → bradycardia inversed / zombie

It still exports audio files for all three ECG types.
"""

import os
import time
import platform
import subprocess
import numpy as np
import scipy.io.wavfile as wavfile

print("=" * 68)
print("  ☣  ZOMBIE ECG PROJECT — generating 4 video-ready DFT figures")
print("  Using REAL MIT-BIH ECG recordings + REAL NSTDB noise where available")
print("=" * 68)

from ecg_signals import (
    generate_normal,
    generate_arrhythmia,
    generate_bradycardia,
    add_real_noise,
    NOISE_DIR,
)
from dft_filter import recover_ecg, heart_rate_dft
from stress_tests import window_sensitivity, noise_robustness, failure_cases
from plots import (
    plot_main_noisy_vs_recovered,
    plot_dft_filter_explanation,
    plot_stress_tests_summary,
    plot_dft_failure_case,
)

FS = 500
DURATION = 10.0
SNR_DB = 5
AUDIO_FS = 44100
OPEN_FIGURES = True   # set to False if you do not want the PNGs to open automatically

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Clear old video outputs so you cannot accidentally use outdated images
# ─────────────────────────────────────────────────────────────────────────────
for filename in os.listdir(FIG_DIR):
    if filename.endswith((".png", ".wav")):
        os.remove(os.path.join(FIG_DIR, filename))
print(f"  Cleared old PNG/WAV outputs from: {FIG_DIR}")


def save_audio(ecg, t, label, out_dir=FIG_DIR):
    """Save a short stereo sonification of one ECG signal."""
    t_audio = np.arange(0, t[-1], 1.0 / AUDIO_FS)
    ecg_audio = np.interp(t_audio, t, ecg)
    ecg_audio = ecg_audio - np.mean(ecg_audio)

    peak = np.max(np.abs(ecg_audio))
    if peak > 1e-9:
        ecg_audio = ecg_audio / peak

    kernel = np.ones(15) / 15
    ecg_smooth = np.convolve(ecg_audio, kernel, mode="same")

    carrier_freq = 220.0
    modulation_depth = 500.0
    phase = 2 * np.pi * (
        carrier_freq * t_audio +
        modulation_depth * np.cumsum(ecg_smooth) / AUDIO_FS
    )
    fm_wave = np.sin(phase)

    derivative = np.abs(np.gradient(ecg_smooth))
    derivative /= np.max(derivative) + 1e-9
    pulse = derivative ** 2
    pulse = np.convolve(pulse, np.hanning(200), mode="same")
    pulse_tone = pulse * np.sin(2 * np.pi * 80 * t_audio)

    audio = 0.75 * fm_wave + 0.55 * pulse_tone
    stereo = np.vstack([audio, np.roll(audio, 120)]).T
    stereo /= np.max(np.abs(stereo)) + 1e-9
    stereo = np.tanh(1.5 * stereo)

    path = os.path.join(out_dir, f"audio_{label}.wav")
    wavfile.write(path, AUDIO_FS, np.int16(stereo * 32767))
    print(f"  ♪  saved {path}")
    return path


def open_file(path):
    """Open a file with the default system app."""
    try:
        if platform.system() == "Windows":
            os.startfile(path)  # noqa: B606 - intentional local file opening
        elif platform.system() == "Darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as exc:
        print(f"  ! Could not open {path}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load clean ECG signals
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading ECG signals ...")

# Required story order: normal → arrhythmia/turning → bradycardia inversed/zombie
t, ecg_normal = generate_normal(duration=DURATION)
_, ecg_arrhythmia = generate_arrhythmia(duration=DURATION)
_, ecg_bradycardia = generate_bradycardia(duration=DURATION)

signals_clean = {
    "normal": ecg_normal,
    "arrhythmia": ecg_arrhythmia,
    "bradycardia": ecg_bradycardia,
}

print(f"   Normal                         BPM (DFT): {heart_rate_dft(ecg_normal, FS):.1f}")
print(f"   Arrhythmia / turning           BPM (DFT): {heart_rate_dft(ecg_arrhythmia, FS):.1f}")
print(f"   Bradycardia inversed / zombie  BPM (DFT): {heart_rate_dft(ecg_bradycardia, FS):.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Add noise and recover with DFT filtering
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2/5] Adding noise and recovering ECGs with DFT filtering ...")
print(f"      Noise directory: {NOISE_DIR}")

signals_noisy = {}
signals_recovered = {}
recovery_info = {}

for key in ["normal", "arrhythmia", "bradycardia"]:
    clean = signals_clean[key]

    noisy = add_real_noise(clean, FS, snr_db=SNR_DB, noise_dir=NOISE_DIR)
    recovered, X_noisy, X_recovered, freqs = recover_ecg(noisy, FS)

    signals_noisy[key] = noisy
    signals_recovered[key] = recovered
    recovery_info[key] = {
        "X_noisy": X_noisy,
        "X_recovered": X_recovered,
        "freqs": freqs,
    }

    print(
        f"   {key:12s} clean={heart_rate_dft(clean, FS):6.1f} BPM   "
        f"noisy={heart_rate_dft(noisy, FS):6.1f} BPM   "
        f"recovered={heart_rate_dft(recovered, FS):6.1f} BPM"
    )

    save_audio(clean,     t, f"{key}_original")
    save_audio(noisy,     t, f"{key}_distorted")
    save_audio(recovered, t, f"{key}_recovered")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generate figures 1 and 2
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Generating main result and DFT explanation figures ...")

plot_main_noisy_vs_recovered(t, signals_noisy, signals_recovered, FS)

# Use normal ECG as the clearest method example. The result figure already
# shows all three ECG types, so the DFT explanation does not need three copies.
plot_dft_filter_explanation(
    recovery_info["normal"]["freqs"],
    recovery_info["normal"]["X_noisy"],
    recovery_info["normal"]["X_recovered"],
    example_label="normal",
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stress tests
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Running stress tests ...")
t0 = time.time()
window_data = window_sensitivity(fs=FS, duration=DURATION)
noise_data = noise_robustness(fs=FS, duration=DURATION)
plot_stress_tests_summary(window_data, noise_data)
print(f"   Stress tests completed in {time.time() - t0:.1f} s")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Failure / limitation case
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Generating DFT limitation / failure case figure ...")
failure_data = failure_cases(fs=FS)
plot_dft_failure_case(failure_data)


# ─────────────────────────────────────────────────────────────────────────────
# Open the final four video figures
# ─────────────────────────────────────────────────────────────────────────────
figures_to_open = [
    "fig1_main_noisy_vs_recovered.png",
    "fig2_dft_filter_explanation.png",
    "fig3_stress_tests_summary.png",
    "fig4_dft_failure_case.png",
]

print("\n" + "=" * 68)
print("  Done. Final video figures saved to:")
for fig_name in figures_to_open:
    fig_path = os.path.join(FIG_DIR, fig_name)
    print(f"   - {fig_path}")
    if OPEN_FIGURES and os.path.exists(fig_path):
        open_file(fig_path)
print("\n  Audio files were also saved in the same figures/ folder.")
print("=" * 68)
