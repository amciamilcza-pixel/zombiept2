"""
run_me.py
=========
ZOMBIE APOCALYPSE ECG — DFT Signals & Systems Project
Signals and Systems 4CA20, 2025-2026
------------------------------------------------------
Loads REAL MIT-BIH Arrhythmia Database recordings.
Uses REAL MIT-BIH Noise Stress Test Database recordings for noise.

REQUIREMENTS
------------
1. pip install numpy matplotlib scipy wfdb
2. Place the mitdb/ folder in the same directory as this file:

   zombieheart/
     mitdb/
       100.dat  100.hea
       116.dat  116.hea  ... etc
     nstdb/
       bw.dat   bw.hea       ← MIT-BIH Noise Stress Test Database
       em.dat   em.hea         (baseline wander, electrode motion,
       ma.dat   ma.hea          muscle artifact)
     run_me.py
     ecg_signals.py
     ...

   !! If your noise files live elsewhere, update NOISE_DIR inside
      ecg_signals.py (clearly marked at the top of that file) !!

Run:
    python3 run_me.py

Output saved to ./figures/
  Plots
  ─────
  fig1_three_hearts.png          three cardiac signatures
  fig2_recovery_normal.png       DFT recovery pipeline, normal
  fig2_recovery_bradycardia.png  DFT recovery pipeline, dying/inverse
  fig2_recovery_arrhythmia.png   DFT recovery pipeline, zombie
  fig3a_window_spectra.png       parameter sensitivity: window spectra
  fig3b_window_accuracy.png      parameter sensitivity: BPM accuracy
  fig4_noise_robustness.png      noise robustness: BPM error vs SNR
  fig5_failure_cases.png         failure cases: leakage + non-stationarity
  fig6_interesting_cases.png     real MIT-BIH interesting records

  Audio  (44100 Hz WAV, ~5 s each, ECG content pitch-shifted into audible range)
  ─────────────────────────────────────────────────────────────────────────────
  audio_normal_original.wav      clean normal sinus rhythm
  audio_normal_distorted.wav     after real noise corruption
  audio_normal_recovered.wav     after DFT bandpass recovery

  audio_bradycardia_original.wav
  audio_bradycardia_distorted.wav
  audio_bradycardia_recovered.wav

  audio_arrhythmia_original.wav
  audio_arrhythmia_distorted.wav
  audio_arrhythmia_recovered.wav
"""

import sys, time, os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample as scipy_resample

print("=" * 60)
print("  ☣  ZOMBIE ECG PROJECT -- generating all figures + audio")
print("  Using REAL MIT-BIH recordings + REAL NSTDB noise")
print("=" * 60)

from ecg_signals  import (generate_normal, generate_bradycardia,
                           generate_arrhythmia, add_real_noise,
                           load_all_interesting_cases, NOISE_DIR)
from dft_filter   import recover_ecg, heart_rate_dft
from stress_tests import (window_sensitivity, noise_robustness,
                           failure_cases)
from plots        import (plot_three_hearts, plot_recovery_pipeline,
                           plot_window_sensitivity, plot_noise_robustness,
                           plot_failure_cases, plot_interesting_cases)

FS       = 500
DURATION = 10.0
SNR_DB   = 5

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED ECG SONIFICATION
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_FS = 44100


def save_audio(ecg, t, label, fs=FS, out_dir=FIG_DIR):
    """
    ECG sonification using:
      1. Frequency modulation (main audible heartbeat)
      2. Pulse enhancement for R-peaks
      3. Stereo ambience for dramatic effect
    """

    # ------------------------------------------------------------------
    # Resample ECG into audio timeline
    # ------------------------------------------------------------------
    t_audio = np.arange(0, t[-1], 1.0 / AUDIO_FS)

    ecg_audio = np.interp(t_audio, t, ecg)

    # Remove DC offset
    ecg_audio = ecg_audio - np.mean(ecg_audio)

    # Normalize safely
    peak = np.max(np.abs(ecg_audio))
    if peak > 1e-9:
        ecg_audio = ecg_audio / peak

    # ------------------------------------------------------------------
    # Smooth slightly (reduces harsh artifacts)
    # ------------------------------------------------------------------
    kernel_size = 15
    kernel = np.ones(kernel_size) / kernel_size
    ecg_smooth = np.convolve(ecg_audio, kernel, mode='same')

    # ------------------------------------------------------------------
    # FM SYNTHESIS
    # ------------------------------------------------------------------
    carrier_freq = 220.0

    # Bigger modulation = clearer heartbeat differences
    modulation_depth = 500.0

    phase = 2 * np.pi * (
        carrier_freq * t_audio +
        modulation_depth * np.cumsum(ecg_smooth) / AUDIO_FS
    )

    fm_wave = np.sin(phase)

    # ------------------------------------------------------------------
    # Add heartbeat pulse clicks from sharp ECG transitions
    # ------------------------------------------------------------------
    derivative = np.abs(np.gradient(ecg_smooth))

    derivative /= np.max(derivative) + 1e-9

    pulse = derivative ** 2

    # Short pulse envelope
    pulse = np.convolve(
        pulse,
        np.hanning(200),
        mode='same'
    )

    pulse_tone = pulse * np.sin(2 * np.pi * 80 * t_audio)

    # ------------------------------------------------------------------
    # Blend layers
    # ------------------------------------------------------------------
    audio = (
        0.75 * fm_wave +
        0.55 * pulse_tone
    )

    # ------------------------------------------------------------------
    # Gentle stereo widening
    # ------------------------------------------------------------------
    left = audio
    right = np.roll(audio, 120)

    stereo = np.vstack([left, right]).T

    # Normalize final output
    stereo /= np.max(np.abs(stereo)) + 1e-9

    # Soft limiting
    stereo = np.tanh(1.5 * stereo)

    audio_int16 = np.int16(stereo * 32767)

    path = os.path.join(out_dir, f"audio_{label}.wav")

    wavfile.write(path, AUDIO_FS, audio_int16)

    print(f"  ♪  saved {path}")

    return path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Load real MIT-BIH recordings
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading real MIT-BIH recordings ...")

t, ecg_normal      = generate_normal     (duration=DURATION)
_, ecg_bradycardia = generate_bradycardia(duration=DURATION)
_, ecg_arrhythmia  = generate_arrhythmia (duration=DURATION)

signals_clean = {
    "normal":      ecg_normal,
    "bradycardia": ecg_bradycardia,
    "arrhythmia":  ecg_arrhythmia,
}

print(f"   Normal       BPM (DFT): {heart_rate_dft(ecg_normal, FS):.1f}")
print(f"   Bradycardia  BPM (DFT): {heart_rate_dft(ecg_bradycardia, FS):.1f}")
print(f"   Arrhythmia   BPM (DFT): {heart_rate_dft(ecg_arrhythmia, FS):.1f}")

plot_three_hearts(t, signals_clean, FS)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Noise corruption → DFT recovery  +  audio export
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2/6] Noise injection (real NSTDB, SNR = {SNR_DB} dB) + DFT recovery ...")
print(f"      Noise dir : {NOISE_DIR}")

for label, ecg_clean, color_key in [
    ("normal",      ecg_normal,      "normal"),
    ("bradycardia", ecg_bradycardia,  "bradycardia"),
    ("arrhythmia",  ecg_arrhythmia,  "arrhythmia"),
]:
    # ── add REAL physiological noise ─────────────────────────────────────────
    noisy = add_real_noise(ecg_clean, FS, snr_db=SNR_DB,
                           noise_dir=NOISE_DIR)

    # ── DFT recovery ─────────────────────────────────────────────────────────
    recovered, X_noisy, X_recovered, freqs = recover_ecg(noisy, FS)

    print(f"   {label:15s}  clean BPM={heart_rate_dft(ecg_clean,  FS):.1f}  "
          f"noisy BPM={heart_rate_dft(noisy,     FS):.1f}  "
          f"recovered BPM={heart_rate_dft(recovered, FS):.1f}")

    # ── recovery pipeline plot ────────────────────────────────────────────────
    plot_recovery_pipeline(t, ecg_clean, noisy, recovered,
                           X_noisy, X_recovered, freqs, FS,
                           label=label, color_key=color_key)

    # ── audio export: original / distorted / recovered ───────────────────────
    save_audio(ecg_clean, t, f"{label}_original")
    save_audio(noisy,     t, f"{label}_distorted")
    save_audio(recovered, t, f"{label}_recovered")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Stress test 1: parameter sensitivity
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Stress test 1 -- parameter sensitivity (window length) ...")
t0 = time.time()
ws_data = window_sensitivity(fs=FS, duration=DURATION)
plot_window_sensitivity(ws_data)
print(f"   Window sizes : {ws_data['window_sizes']}")
print(f"   BPM estimates: {[f'{b:.1f}' for b in ws_data['hr_estimates']]}")
print(f"   ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Stress test 2: noise robustness
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Stress test 2 -- noise robustness (SNR sweep) ...  ", end="", flush=True)
t0 = time.time()
nr_data = noise_robustness(fs=FS, duration=DURATION)
plot_noise_robustness(nr_data)
print(f"done  ({time.time()-t0:.1f}s)")
for name, d in nr_data.items():
    errors = np.array(d["hr_error"])
    snrs   = np.array(d["snr_levels"])
    bad    = snrs[errors > 10]
    if len(bad):
        print(f"   {name:15s}: detection breaks below SNR ~ {bad.max():.1f} dB")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Stress test 3: failure cases
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Stress test 3 -- failure cases ...")
fc_data = failure_cases(fs=FS)
plot_failure_cases(fc_data)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Interesting MIT-BIH cases
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Loading interesting MIT-BIH cases ...")
cases = load_all_interesting_cases(duration=DURATION)
if cases:
    plot_interesting_cases(cases, fs=FS)
else:
    print("  ! No cases loaded. Check your mitdb/ folder.")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Done! All figures + 9 audio files saved to ./figures/")
print("  Audio files: 3 heart types × 3 versions (original / distorted / recovered)")
print("  Note: ECG frequencies are shifted ×88 into the audible range.")
print("        Heartbeat rhythm is clearly distinguishable across types.")
print("=" * 60)