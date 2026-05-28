"""
plots.py
--------
All plotting functions for the Zombie Apocalypse ECG / DFT project.
Each function saves a figure to figures/ and returns the Figure object.

Style: dark 'apocalypse' theme (dark background, red / green / amber accents)
so the video looks distinctive and coherent.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — safe on all machines
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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
    "normal":       "#00ff88",   # green  — survivor
    "bradycardia":  "#4488ff",   # blue   — turning
    "arrhythmia":   "#ff3333",   # red    — zombie
    "noisy":        "#888888",
    "recovered":    "#ffcc00",
    "accent":       "#ff6600",
}

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓  saved {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Three heart types: time domain + spectrum
# ─────────────────────────────────────────────────────────────────────────────

def plot_three_hearts(t, signals, fs, title_suffix=""):
    """
    signals : dict  {"normal": array, "bradycardia": array, "arrhythmia": array}
    Shows time domain (left) and DFT magnitude (right) for all three.
    """
    labels = {
        "normal":      "NORMAL  (~70 BPM)",
        "bradycardia": "BRADYCARDIA  (~30 BPM)  [dying]",
        "arrhythmia":  "ARRHYTHMIA  (~140 BPM)  [zombie]",
    }
    keys = ["normal", "bradycardia", "arrhythmia"]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"☣  ZOMBIE APOCALYPSE ECG — Three Cardiac Signatures  {title_suffix}",
                 fontsize=13, color=COLORS["accent"], y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    for row, key in enumerate(keys):
        sig = signals[key]
        color = COLORS[key]

        # Time domain
        ax_t = fig.add_subplot(gs[row, 0])
        ax_t.plot(t[:int(5*fs)], sig[:int(5*fs)], color=color, lw=1.2)
        ax_t.set_title(f"{labels[key]}  — time domain", fontsize=9, color=color)
        ax_t.set_xlabel("Time (s)", fontsize=8)
        ax_t.set_ylabel("Amplitude", fontsize=8)
        ax_t.grid(True)

        # DFT magnitude (0–10 Hz, cardiac region)
        N = len(sig)
        X = np.fft.fft(sig)
        freqs = np.fft.fftfreq(N, d=1.0/fs)
        half = N // 2
        f_pos = freqs[:half]
        m_pos = np.abs(X[:half])

        ax_f = fig.add_subplot(gs[row, 1])
        mask = f_pos <= 10
        ax_f.plot(f_pos[mask], m_pos[mask], color=color, lw=1.2)
        ax_f.set_title(f"{labels[key]}  — DFT magnitude", fontsize=9, color=color)
        ax_f.set_xlabel("Frequency (Hz)", fontsize=8)
        ax_f.set_ylabel("|X(f)|", fontsize=8)
        ax_f.grid(True)

    _save(fig, "fig1_three_hearts.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Noise corruption + DFT recovery pipeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_recovery_pipeline(t, ecg_clean, ecg_noisy, ecg_recovered,
                           X_noisy, X_recovered, freqs, fs, label, color_key):
    """
    Four-panel figure showing the full DFT recovery pipeline for one heart type.
      Panel 1: clean ECG (ground truth)
      Panel 2: noisy signal (apocalypse corruption)
      Panel 3: DFT of noisy vs. DFT after filtering
      Panel 4: recovered ECG
    """
    color = COLORS[color_key]
    N = len(freqs)
    half = N // 2
    f_pos = freqs[:half]

    # One-sided magnitude
    m_noisy = np.abs(X_noisy[:half])
    m_recov = np.abs(X_recovered[:half])

    display_samples = int(5 * fs)   # show 5 seconds

    fig, axes = plt.subplots(4, 1, figsize=(13, 11))
    fig.suptitle(f"☣  DFT RECOVERY PIPELINE — {label.upper()}",
                 fontsize=12, color=COLORS["accent"])

    # 1) Clean
    axes[0].plot(t[:display_samples], ecg_clean[:display_samples],
                 color=color, lw=1.3)
    axes[0].set_title("[1] Clean ECG  (ground truth)", fontsize=9, color=color)
    axes[0].set_ylabel("Amplitude"); axes[0].grid(True)

    # 2) Noisy
    axes[1].plot(t[:display_samples], ecg_noisy[:display_samples],
                 color=COLORS["noisy"], lw=0.8, alpha=0.9)
    axes[1].set_title("[2] Apocalypse noise added  (SNR = 5 dB)", fontsize=9,
                      color=COLORS["noisy"])
    axes[1].set_ylabel("Amplitude"); axes[1].grid(True)

    # 3) DFT comparison
    f_mask = f_pos <= 60
    axes[2].plot(f_pos[f_mask], m_noisy[f_mask],
                 color=COLORS["noisy"], lw=1.0, alpha=0.7, label="Noisy DFT")
    axes[2].plot(f_pos[f_mask], m_recov[f_mask],
                 color=COLORS["recovered"], lw=1.3, label="After DFT filter")
    axes[2].set_title("[3] DFT magnitude: before vs. after filtering", fontsize=9,
                      color=COLORS["recovered"])
    axes[2].set_xlabel("Frequency (Hz)"); axes[2].set_ylabel("|X(f)|")
    axes[2].legend(fontsize=8, facecolor="#1a1a1a"); axes[2].grid(True)

    # 4) Recovered
    axes[3].plot(t[:display_samples], ecg_recovered[:display_samples],
                 color=COLORS["recovered"], lw=1.3)
    axes[3].plot(t[:display_samples], ecg_clean[:display_samples],
                 color=color, lw=0.8, alpha=0.5, label="Original (ref)")
    axes[3].set_title("[4] Recovered ECG  (inverse DFT)", fontsize=9,
                      color=COLORS["recovered"])
    axes[3].set_xlabel("Time (s)"); axes[3].set_ylabel("Amplitude")
    axes[3].legend(fontsize=8, facecolor="#1a1a1a"); axes[3].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, f"fig2_recovery_{color_key}.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Stress test 1: Window / parameter sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_window_sensitivity(data):
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("☣  STRESS TEST 1 — DFT Window Length Sensitivity",
                 fontsize=12, color=COLORS["accent"])

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4)

    spectra      = data["spectra"]
    window_sizes = data["window_sizes"]
    fs           = data["fs"]

    for i, (N, (freqs, mag)) in enumerate(zip(window_sizes[:6], spectra[:6])):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        mask = freqs <= 10
        ax.plot(freqs[mask], mag[mask], color=COLORS["normal"], lw=1.2)
        df = fs / N
        ax.set_title(f"N = {N}   Δf = {df:.2f} Hz", fontsize=9,
                     color=COLORS["normal"])
        ax.set_xlabel("Freq (Hz)", fontsize=7)
        ax.set_ylabel("|X(f)|", fontsize=7)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig3a_window_spectra.png")

    # BPM accuracy vs window
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("☣  Window Length: Δf and BPM Accuracy", color=COLORS["accent"])

    axes2[0].plot(window_sizes, data["delta_f"], "o-", color=COLORS["accent"])
    axes2[0].set_xlabel("Window N (samples)"); axes2[0].set_ylabel("Δf (Hz)")
    axes2[0].set_title("Frequency Resolution vs N", color=COLORS["accent"])
    axes2[0].invert_xaxis(); axes2[0].grid(True)

    axes2[1].axhline(data["true_bpm"], color=COLORS["normal"],
                     lw=1.5, linestyle="--", label="True BPM = 70")
    axes2[1].plot(window_sizes, data["hr_estimates"], "o-",
                  color=COLORS["recovered"], label="DFT estimate")
    axes2[1].set_xlabel("Window N (samples)"); axes2[1].set_ylabel("BPM")
    axes2[1].set_title("Heart Rate Estimate vs N", color=COLORS["recovered"])
    axes2[1].legend(fontsize=8, facecolor="#1a1a1a"); axes2[1].grid(True)

    plt.tight_layout()
    _save(fig2, "fig3b_window_accuracy.png")
    return fig, fig2


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Stress test 2: Noise robustness
# ─────────────────────────────────────────────────────────────────────────────

def plot_noise_robustness(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("☣  STRESS TEST 2 — Noise Robustness: BPM Error vs SNR",
                 fontsize=12, color=COLORS["accent"])

    name_labels = {
        "normal":      ("NORMAL", COLORS["normal"]),
        "bradycardia": ("BRADYCARDIA", COLORS["bradycardia"]),
        "arrhythmia":  ("ARRHYTHMIA", COLORS["arrhythmia"]),
    }

    for ax, (name, (label, color)) in zip(axes, name_labels.items()):
        d = results[name]
        ax.axhline(10, color="#ff6600", lw=1, linestyle="--",
                   label="10 BPM error threshold")
        ax.plot(d["snr_levels"], d["hr_error"], "o-", color=color, lw=1.5)
        ax.fill_between(d["snr_levels"], d["hr_error"],
                        alpha=0.15, color=color)
        ax.set_title(f"{label}\n(true = {d['true_bpm']:.0f} BPM)",
                     fontsize=10, color=color)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("|BPM error|")
        ax.legend(fontsize=7, facecolor="#1a1a1a"); ax.grid(True)
        ax.invert_xaxis()   # high SNR on right (easier → harder left to right)

    plt.tight_layout()
    _save(fig, "fig4_noise_robustness.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Stress test 3: Failure cases
# ─────────────────────────────────────────────────────────────────────────────

def plot_failure_cases(data):
    leak = data["leakage"]
    ns   = data["nonstationarity"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("☣  STRESS TEST 3 — DFT Failure Cases",
                 fontsize=12, color=COLORS["accent"])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── A1: time segment with leakage ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(leak["t_seg"], leak["seg"], color=COLORS["normal"], lw=1.2)
    ax1.set_title(f"Failure A: Short segment  (N={leak['N']})\n"
                  "Non-integer periods → leakage",
                  fontsize=9, color=COLORS["normal"])
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude"); ax1.grid(True)

    # ── A2: leakage comparison ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    mask = leak["freqs"] <= 8
    ax2.plot(leak["freqs"][mask], leak["mag_rect"][mask],
             color="#ff3333", lw=1.2, alpha=0.85, label="Rectangular window")
    ax2.plot(leak["freqs"][mask], leak["mag_hann"][mask],
             color=COLORS["recovered"], lw=1.2, label="Hann window")
    ax2.set_title("Spectral leakage: rectangular vs Hann window",
                  fontsize=9, color=COLORS["recovered"])
    ax2.set_xlabel("Freq (Hz)"); ax2.set_ylabel("|X(f)|")
    ax2.legend(fontsize=8, facecolor="#1a1a1a"); ax2.grid(True)

    # ── B1: global DFT of arrhythmia ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    mask_g = ns["freqs_global"] <= 6
    ax3.plot(ns["freqs_global"][mask_g], ns["mag_global"][mask_g],
             color=COLORS["arrhythmia"], lw=1.2)
    ax3.set_title("Failure B: Global DFT of arrhythmia\n"
                  "Looks like one frequency — misleading!",
                  fontsize=9, color=COLORS["arrhythmia"])
    ax3.set_xlabel("Freq (Hz)"); ax3.set_ylabel("|X(f)|"); ax3.grid(True)

    # ── B2: sliding-window DFT (STFT) reveals non-stationarity ───────────────
    ax4 = fig.add_subplot(gs[1, 1])
    freq_mask = ns["stft_freqs"] <= 5
    im = ax4.pcolormesh(
        ns["stft_times"],
        ns["stft_freqs"][freq_mask],
        ns["stft_mag"][freq_mask, :],
        cmap="inferno", shading="gouraud"
    )
    plt.colorbar(im, ax=ax4, label="Magnitude")
    ax4.set_title("Sliding-window DFT reveals time-varying frequency\n"
                  "(what global DFT hides)",
                  fontsize=9, color=COLORS["accent"])
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Freq (Hz)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig5_failure_cases.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Interesting MIT-BIH cases
# ─────────────────────────────────────────────────────────────────────────────

def plot_interesting_cases(cases: dict, fs: int = 500):
    """
    cases : dict  record_id -> (t, sig, meta)
    Shows time domain + DFT for each interesting record.
    """
    n = len(cases)
    if n == 0:
        print("  ! No interesting cases loaded — skipping figure 6")
        return None

    fig = plt.figure(figsize=(16, 3.2 * n))
    fig.suptitle("☣  MIT-BIH INTERESTING CASES — Real Clinical ECG Records",
                 fontsize=12, color=COLORS["accent"], y=0.99)

    gs = gridspec.GridSpec(n, 2, figure=fig, hspace=0.7, wspace=0.35)

    for row, (rec_id, (t, sig, meta)) in enumerate(cases.items()):
        color = meta.get('color', '#ffffff')
        label = meta.get('label', rec_id)
        desc  = meta.get('description', '')

        display = int(5 * fs)

        # ── time domain ──────────────────────────────────────────────────────
        ax_t = fig.add_subplot(gs[row, 0])
        ax_t.plot(t[:display], sig[:display], color=color, lw=1.0)
        ax_t.set_title(label, fontsize=9, color=color)
        ax_t.set_xlabel("Time (s)", fontsize=7)
        ax_t.set_ylabel("mV", fontsize=7)
        ax_t.grid(True)

        # Description as text box
        ax_t.text(0.98, 0.97, desc,
                  transform=ax_t.transAxes,
                  fontsize=6.5, color="#aaaaaa",
                  va='top', ha='right',
                  bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='#1a1a1a', alpha=0.8, edgecolor='none'))

        # ── DFT magnitude (0–10 Hz) ───────────────────────────────────────────
        N = len(sig)
        X = np.fft.fft(sig)
        freqs = np.fft.fftfreq(N, d=1.0 / fs)
        half  = N // 2
        f_pos = freqs[:half]
        m_pos = np.abs(X[:half])

        ax_f = fig.add_subplot(gs[row, 1])
        mask = f_pos <= 10
        ax_f.plot(f_pos[mask], m_pos[mask], color=color, lw=1.0)
        ax_f.set_title(f"{label} — DFT", fontsize=9, color=color)
        ax_f.set_xlabel("Frequency (Hz)", fontsize=7)
        ax_f.set_ylabel("|X(f)|", fontsize=7)
        ax_f.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, "fig6_interesting_cases.png")
    return fig
