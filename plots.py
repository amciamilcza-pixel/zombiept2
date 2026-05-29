"""
plots.py
--------
Presentation-focused plotting functions for the Zombie Apocalypse ECG / DFT project.

Only four video-ready figures are generated:
  fig1_main_noisy_vs_recovered.png
  fig2_dft_filter_explanation.png
  fig3_stress_tests_summary.png
  fig4_dft_failure_case.png

Story order everywhere:
  1. normal
  2. arrhythmia / turning
  3. bradycardia inversed / zombie
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe backend; saves figures without needing a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d0d0d",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#cccccc",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "text.color":        "#cccccc",
    "grid.color":        "#2a2a2a",
    "grid.linestyle":    "--",
    "lines.linewidth":   1.4,
    "font.family":       "monospace",
})

COLORS = {
    "normal":      "#00ff88",  # green  — normal survivor
    "arrhythmia":  "#ff6600",  # orange — turning / infected
    "bradycardia": "#ff3333",  # red    — inverted zombie
    "noisy":       "#888888",
    "recovered":   "#ffcc00",
    "accent":      "#ff6600",
}

LABELS = {
    "normal":      "NORMAL",
    "arrhythmia":  "ARRHYTHMIA / TURNING",
    "bradycardia": "BRADYCARDIA INVERSED / ZOMBIE",
}

ORDER = ["normal", "arrhythmia", "bradycardia"]

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    """Save a figure into the local figures/ folder and print exact path."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓  saved {path}")
    return path


def _one_sided_spectrum(x, fs):
    """Return positive frequencies and one-sided DFT magnitude."""
    N = len(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    half = N // 2
    return freqs[:half], np.abs(X[:half])


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Main result: noisy vs recovered for all three cases
# ─────────────────────────────────────────────────────────────────────────────

def plot_main_noisy_vs_recovered(t, noisy_signals, recovered_signals, fs):
    """
    Video-ready main result.

    One row per ECG case. Each row overlays:
      - noisy signal in grey
      - recovered signal in the case color

    No separate clean/original trace is shown, so the figure stays readable.
    """
    display = int(5 * fs)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "☣  ECG RECOVERY RESULT — Noisy Signal vs DFT-Recovered Signal",
        fontsize=13, color=COLORS["accent"], y=0.98
    )

    for ax, key in zip(axes, ORDER):
        ax.plot(t[:display], noisy_signals[key][:display],
                color=COLORS["noisy"], lw=0.8, alpha=0.75, label="Noisy ECG")
        ax.plot(t[:display], recovered_signals[key][:display],
                color=COLORS[key], lw=1.4, label="Recovered ECG")

        ax.set_title(LABELS[key], fontsize=10, color=COLORS[key])
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a1a")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig1_main_noisy_vs_recovered.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — DFT filtering explanation: one clear example
# ─────────────────────────────────────────────────────────────────────────────

def plot_dft_filter_explanation(freqs, X_noisy, X_recovered,
                                example_label="normal", max_freq=80):
    """
    Show why the DFT is essential: filtering is done by changing the spectrum.

    This intentionally uses one example only, not all three cases, to avoid
    overcrowding the video.
    """
    half = len(freqs) // 2
    f_pos = freqs[:half]
    mag_noisy = np.abs(X_noisy[:half])
    mag_recovered = np.abs(X_recovered[:half])

    mask = (f_pos >= 0) & (f_pos <= max_freq)
    f = f_pos[mask]
    mn = mag_noisy[mask]
    mr = mag_recovered[mask]

    # Normalise only for display clarity; this does not change the processing.
    scale = max(np.max(mn), np.max(mr), 1e-12)
    mn = mn / scale
    mr = mr / scale

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.suptitle(
        "☣  DFT FILTERING EXPLANATION — What the Recovery Removes and Keeps",
        fontsize=13, color=COLORS["accent"]
    )

    ax.plot(f, mn, color=COLORS["noisy"], lw=1.0, alpha=0.75, label="Noisy DFT")
    ax.plot(f, mr, color=COLORS["recovered"], lw=1.5, label="After DFT filtering")

    # Visual explanation regions. These are explanatory annotations; the exact
    # recovery parameters are implemented in dft_filter.py.
    ax.axvspan(0.5, 40, color=COLORS["normal"], alpha=0.08,
               label="ECG band kept")
    ax.axvline(50, color=COLORS["bradycardia"], lw=1.2, linestyle="--",
               label="50 Hz hum removed")

    ax.text(50.8, 0.92, "50 Hz\nnotch", color=COLORS["bradycardia"], fontsize=8,
            va="top")
    ax.text(2, 0.12, "main ECG frequency content", color=COLORS["normal"], fontsize=8)

    ax.set_title(f"Example signal: {LABELS.get(example_label, example_label)}",
                 fontsize=10, color=COLORS.get(example_label, COLORS["accent"]))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalised |X(f)|")
    ax.set_xlim(0, max_freq)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(fontsize=8, facecolor="#1a1a1a", loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "fig2_dft_filter_explanation.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Stress test summary: parameter sensitivity + noise robustness
# ─────────────────────────────────────────────────────────────────────────────

def plot_stress_tests_summary(window_data, noise_data):
    """
    Combined video figure for two required stress tests:
      left  = DFT parameter sensitivity via window length / frequency resolution
      right = noise robustness via BPM error vs SNR
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        "☣  DFT STRESS TESTS — Parameter Sensitivity and Noise Robustness",
        fontsize=13, color=COLORS["accent"]
    )

    # ── left: window length sensitivity ──────────────────────────────────────
    ax = axes[0]
    window_sizes = np.array(window_data["window_sizes"])
    hr_estimates = np.array(window_data["hr_estimates"])
    true_bpm = float(window_data["true_bpm"])

    ax.axhline(true_bpm, color=COLORS["normal"], lw=1.3, linestyle="--",
               label=f"Reference BPM ≈ {true_bpm:.0f}")
    ax.plot(window_sizes, hr_estimates, "o-", color=COLORS["recovered"], lw=1.5,
            label="DFT BPM estimate")

    # Optional secondary axis for Δf if available.
    if "delta_f" in window_data:
        ax2 = ax.twinx()
        ax2.plot(window_sizes, window_data["delta_f"], "s--",
                 color=COLORS["accent"], lw=1.0, alpha=0.8, label="Δf = fs/N")
        ax2.set_ylabel("Frequency resolution Δf (Hz)", color=COLORS["accent"])
        ax2.tick_params(axis="y", colors=COLORS["accent"])

    ax.set_title("Stress test 1: window length sensitivity", fontsize=10,
                 color=COLORS["recovered"])
    ax.set_xlabel("Window length N (samples)")
    ax.set_ylabel("Estimated BPM")
    ax.grid(True)
    ax.legend(fontsize=8, facecolor="#1a1a1a", loc="best")

    # ── right: noise robustness ──────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(10, color=COLORS["accent"], lw=1.2, linestyle="--",
               label="10 BPM error threshold")

    for key in ORDER:
        if key not in noise_data:
            continue
        d = noise_data[key]
        ax.plot(d["snr_levels"], d["hr_error"], "o-",
                color=COLORS[key], lw=1.5, label=LABELS[key])

    ax.set_title("Stress test 2: noise robustness", fontsize=10,
                 color=COLORS["accent"])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("|BPM error|")
    ax.grid(True)
    ax.invert_xaxis()  # visually goes from cleaner to harder as you move right/left
    ax.legend(fontsize=8, facecolor="#1a1a1a", loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "fig3_stress_tests_summary.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Failure case: global DFT vs sliding-window DFT
# ─────────────────────────────────────────────────────────────────────────────

def plot_dft_failure_case(failure_data):
    """
    Show the limitation of a global DFT for non-stationary / irregular ECG.

    Left: one global DFT spectrum.
    Right: sliding-window DFT / STFT showing time variation.
    """
    ns = failure_data["nonstationarity"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        "☣  DFT LIMITATION — A Global Spectrum Can Hide Time Variation",
        fontsize=13, color=COLORS["accent"]
    )

    # ── global DFT ───────────────────────────────────────────────────────────
    ax = axes[0]
    mask = ns["freqs_global"] <= 6
    ax.plot(ns["freqs_global"][mask], ns["mag_global"][mask],
            color=COLORS["arrhythmia"], lw=1.4)
    ax.set_title("Global DFT of arrhythmia / turning signal", fontsize=10,
                 color=COLORS["arrhythmia"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|X(f)|")
    ax.grid(True)
    ax.text(0.04, 0.94,
            "One spectrum for the whole window\ncan suggest one dominant rhythm.",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      alpha=0.85, edgecolor="none"))

    # ── STFT / sliding-window DFT ────────────────────────────────────────────
    ax = axes[1]
    freq_mask = ns["stft_freqs"] <= 5
    im = ax.pcolormesh(
        ns["stft_times"],
        ns["stft_freqs"][freq_mask],
        ns["stft_mag"][freq_mask, :],
        cmap="inferno",
        shading="gouraud",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Magnitude")
    ax.set_title("Sliding-window DFT reveals changing frequency", fontsize=10,
                 color=COLORS["accent"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.text(0.04, 0.94,
            "The rhythm is not stationary,\nso frequency content changes over time.",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      alpha=0.85, edgecolor="none"))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "fig4_dft_failure_case.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible wrappers
# These prevent older run_me.py versions from crashing, but the new run_me.py
# below uses the four cleaner plotting functions above.
# ─────────────────────────────────────────────────────────────────────────────

def plot_three_hearts(t, signals, fs, title_suffix=""):
    """Old name kept only for compatibility. Prefer plot_main_noisy_vs_recovered."""
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"☣  ECG CLEAN SIGNALS — Three Cardiac Signatures {title_suffix}",
                 fontsize=13, color=COLORS["accent"], y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    for row, key in enumerate(ORDER):
        sig = signals[key]
        color = COLORS[key]
        ax_t = fig.add_subplot(gs[row, 0])
        ax_t.plot(t[:int(5 * fs)], sig[:int(5 * fs)], color=color, lw=1.2)
        ax_t.set_title(f"{LABELS[key]} — clean time domain", fontsize=9, color=color)
        ax_t.set_xlabel("Time (s)"); ax_t.set_ylabel("Amplitude"); ax_t.grid(True)

        f_pos, mag = _one_sided_spectrum(sig, fs)
        mask = f_pos <= 10
        ax_f = fig.add_subplot(gs[row, 1])
        ax_f.plot(f_pos[mask], mag[mask], color=color, lw=1.2)
        ax_f.set_title(f"{LABELS[key]} — DFT magnitude", fontsize=9, color=color)
        ax_f.set_xlabel("Frequency (Hz)"); ax_f.set_ylabel("|X(f)|"); ax_f.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig_unused_clean_signals.png")
    return fig


def plot_recovery_pipeline(*args, **kwargs):
    raise RuntimeError(
        "plot_recovery_pipeline() was removed from the video output set. "
        "Use plot_main_noisy_vs_recovered() and plot_dft_filter_explanation() instead."
    )


def plot_window_sensitivity(data):
    return plot_stress_tests_summary(data, {})


def plot_noise_robustness(results):
    # Minimal compatibility figure if called alone.
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in ORDER:
        if key in results:
            d = results[key]
            ax.plot(d["snr_levels"], d["hr_error"], "o-",
                    color=COLORS[key], label=LABELS[key])
    ax.axhline(10, color=COLORS["accent"], linestyle="--", label="10 BPM threshold")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("|BPM error|"); ax.grid(True)
    ax.legend(fontsize=8, facecolor="#1a1a1a")
    ax.invert_xaxis()
    _save(fig, "fig_unused_noise_robustness.png")
    return fig


def plot_failure_cases(data):
    return plot_dft_failure_case(data)


def plot_interesting_cases(cases: dict, fs: int = 500):
    print("  ! Interesting cases figure skipped for the 4-figure video output set.")
    return None
