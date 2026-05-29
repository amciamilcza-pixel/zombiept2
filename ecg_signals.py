import os
import numpy as np
import wfdb
from scipy.signal import resample as scipy_resample

#sets a path to MIT-BIH database files
NOISE_DIR = os.path.join(os.path.dirname(os.path.abspath("nstdb")), "nstdb")

#helper function that given record ID finds MIT-BIH file (saved with domain .hea)
def _find_mitdb(record_id: str) -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(this_dir, "mitdb", record_id),          # zombieheart/mitdb/
        os.path.join(this_dir, "..", "mitdb", record_id),     # one level up
        os.path.join(os.getcwd(), "mitdb", record_id),        # cwd/mitdb/
        os.path.join(os.getcwd(), record_id),                 # cwd/ directly
    ]
    for path in candidates:
        if os.path.exists(path + ".hea"):
            return path
    raise FileNotFoundError(
        f"\n\n  Could not find MIT-BIH record '{record_id}'.\n"
        f"  Expected location: {os.path.join(this_dir, 'mitdb', record_id + '.hea')}\n"
    )

def _load_record(record_id: str, duration: float = 10.0,
                 start_sec: float = 5.0, channel: int = 0):
    """
    Load `duration` seconds of a MIT-BIH record starting at `start_sec`.
    Resamples to 500 Hz if the record has a different sampling rate.

    Parameters
    ----------
    record_id  : e.g. '100', '207'
    duration   : seconds to extract
    start_sec  : skip the first few seconds (avoids lead-on artefacts)
    channel    : 0 = MLII (standard), 1 = V5 or V1 depending on record

    Returns
    -------
    t   : time axis (seconds, starts at 0)
    sig : ECG signal array
    fs  : effective sample rate (always 500 Hz after resampling)
    """
    path = _find_mitdb(record_id)
    rec  = wfdb.rdrecord(path)
    fs_orig = rec.fs
    sig_all = rec.p_signal[:, channel].astype(float)

    # Replace NaN (missing samples in some records) with linear interpolation
    nans = np.isnan(sig_all)
    if nans.any():
        idx = np.arange(len(sig_all))
        sig_all[nans] = np.interp(idx[nans], idx[~nans], sig_all[~nans])

    # Extract the requested window
    start_samp = int(start_sec * fs_orig)
    end_samp   = start_samp + int(duration * fs_orig)
    end_samp   = min(end_samp, len(sig_all))
    sig        = sig_all[start_samp:end_samp]

    # Resample to 500 Hz using linear interpolation if needed
    target_fs = 500
    if fs_orig != target_fs:
        n_out = int(len(sig) * target_fs / fs_orig)
        t_in  = np.linspace(0, 1, len(sig))
        t_out = np.linspace(0, 1, n_out)
        sig   = np.interp(t_out, t_in, sig)

    # Remove DC offset (centre on zero)
    sig = sig - np.mean(sig)

    t = np.arange(len(sig)) / target_fs
    return t, sig, target_fs


# ── public generators ─────────────────────────────────────────────────────────

def generate_normal(fs=500, duration=10.0, seed=0):
    """
    Record 100 — classic normal sinus rhythm (~75 BPM).
    The cleanest, most regular heartbeat in the MIT-BIH database.
    Used as the healthy survivor baseline.
    """
    t, sig, _ = _load_record('100', duration=duration)
    return t, sig


def generate_bradycardia(fs=500, duration=10.0, seed=1):
    """
    Record 119 — BRADYCARDIA INVERSED / ZOMBIE.
    This is the final zombie stage in the project storyline.

    The signal is inverted (×-1), so the R-peak points DOWN and the
    P/T waves point UP. The amplitude is also scaled to 50% to make
    it look weaker and less alive.
    """
    t, sig, _ = _load_record('119', duration=duration)
    sig_inverted = -sig * 0.5
    return t, sig_inverted


def generate_arrhythmia(fs=500, duration=10.0, seed=2):
    """
    Record 203 — ARRHYTHMIA / TURNING.
    This is the middle stage in the project storyline: the patient is
    turning / infected, so the rhythm is irregular and unstable.

    The signal is not inverted. It remains a chaotic arrhythmia case,
    which also demonstrates that a global DFT can be misleading for
    non-stationary ECG signals.
    """
    t, sig, _ = _load_record('203', duration=duration)
    return t, sig


# ── interesting cases ─────────────────────────────────────────────────────────

INTERESTING_CASES = {
    '207': {
        'label': '207 — Deep S-wave groove',
        'description': 'Pronounced negative deflection after R-peak.\n'
                       'The S-wave is unusually deep — visible as a\n'
                       'strong low-frequency component in the DFT.',
        'color': '#ff9944',
    },
    '203': {
        'label': '203 — Arrhythmia / turning rhythm',
        'description': 'Extremely irregular beat timing and amplitude.\n'
                       'DFT shows a smeared spectrum — no clean harmonics.\n'
                       'Best example of DFT non-stationarity failure.',
        'color': '#ff6600',
    },
    '116': {
        'label': '116 — Very fast ventricular rate',
        'description': 'High heart rate with rapid successive beats.\n'
                       'DFT fundamental shifts to higher frequency.\n'
                       'Harmonics spaced further apart.',
        'color': '#ff6688',
    },
    '118': {
        'label': '118 — Noisy recording',
        'description': 'Heavy baseline wander and electrode noise.\n'
                       'Shows the DFT filter working hardest — the\n'
                       'recovered signal is visibly reconstructed.',
        'color': '#aaaaff',
    },
    '119': {
        'label': '119 — Bradycardia inversed / zombie',
        'description': 'Slow irregular rhythm with conduction blocks.\n'
                       'Used as the dying heart base signal.',
        'color': '#ff3333',
    },
    '122': {
        'label': '122 — Strong sharp peaks, fast',
        'description': 'Very tall R-peaks with fast rate.\n'
                       'DFT shows high-amplitude harmonics extending\n'
                       'further into high frequencies than normal.',
        'color': '#44ffaa',
    },
    '200': {
        'label': '200 — Mixed / weird morphology',
        'description': 'Contains multiple rhythm types in one recording.\n'
                       'DFT shows a complex spectrum — illustrates why\n'
                       'global DFT misleads on non-stationary signals.',
        'color': '#ffcc00',
    },
}


def load_interesting_case(record_id: str, duration: float = 10.0):
    """
    Load one of the interesting case records.
    Returns (t, sig, meta_dict).
    """
    meta = INTERESTING_CASES.get(record_id, {
        'label': record_id,
        'description': '',
        'color': '#ffffff',
    })
    t, sig, fs = _load_record(record_id, duration=duration)
    return t, sig, meta


def load_all_interesting_cases(duration: float = 10.0):
    """
    Load all interesting cases. Returns dict: record_id -> (t, sig, meta).
    Skips any records that can't be found with a warning.
    """
    results = {}
    for rec_id in INTERESTING_CASES:
        try:
            t, sig, meta = load_interesting_case(rec_id, duration=duration)
            results[rec_id] = (t, sig, meta)
            print(f"  ✓  loaded record {rec_id}  ({meta['label']})")
        except FileNotFoundError as e:
            print(f"  ✗  skipped {rec_id} — file not found")
    return results


# ── noise ─────────────────────────────────────────────────────────────────────

def _load_noise_record(name: str, n_samples: int, fs: int,
                        noise_dir: str, seed: int = 0) -> np.ndarray:
    """
    Load `n_samples` samples from a MIT-BIH NSTDB noise record (bw / em / ma).
    The record is at 360 Hz; it is resampled to `fs` Hz on the fly.
    A random offset is chosen so repeated calls give different segments.
    Returns a zero-mean unit-RMS noise array of length `n_samples`.
    """
    path = os.path.join(noise_dir, name)
    rec  = wfdb.rdrecord(path)
    raw  = rec.p_signal[:, 0].astype(float)   # channel 0 only
    fs_orig = rec.fs                           # 360 Hz

    # Resample to target fs using linear interpolation
    n_resampled = int(len(raw) * fs / fs_orig)
    raw_rs = np.interp(
        np.linspace(0, 1, n_resampled),
        np.linspace(0, 1, len(raw)),
        raw,
    )

    # Random offset so each call draws a different chunk
    rng = np.random.default_rng(seed)
    if len(raw_rs) >= n_samples:
        start = rng.integers(0, len(raw_rs) - n_samples)
        chunk = raw_rs[start : start + n_samples]
    else:
        # Tile if signal shorter than requested
        repeats = int(np.ceil(n_samples / len(raw_rs)))
        chunk   = np.tile(raw_rs, repeats)[:n_samples]

    # Normalise to zero-mean, unit RMS
    chunk -= np.mean(chunk)
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms > 1e-10:
        chunk /= rms
    return chunk


def add_real_noise(ecg, fs=500, snr_db=5, seed=99,
                   noise_dir: str = None,
                   weights=(0.45, 0.35, 0.20)):
    """
    Add REAL physiological noise from the MIT-BIH Noise Stress Test Database.

    Three noise sources are mixed:
      bw  (baseline wander)      weight 0.45   — slow electrode drift
      em  (electrode motion)     weight 0.35   — movement artefact
      ma  (muscle artifact)      weight 0.20   — EMG interference

    The combined noise is scaled so the ADDED noise achieves `snr_db` dB
    relative to the clean ECG.  A 50 Hz power-line hum tone is also added
    (25 % of ECG RMS) — this is synthetic because the NSTDB does not
    include a dedicated mains-hum track, and it lets the DFT notch filter
    demonstrate precise frequency-domain removal.

    Parameters
    ----------
    ecg       : clean ECG array (mV, any length)
    fs        : sample rate of `ecg` (Hz)
    snr_db    : signal-to-noise ratio of the ADDED noise (dB)
    seed      : random seed for reproducible segment selection
    noise_dir : folder with bw/em/ma .dat/.hea files.
                Defaults to NOISE_DIR (set at module top).
    weights   : mixing weights for (bw, em, ma) — must sum ≤ 1

    Falls back to add_noise() if the noise files are not found.
    """
    if noise_dir is None:
        noise_dir = NOISE_DIR

    N       = len(ecg)
    t       = np.arange(N) / fs
    sig_rms = np.sqrt(np.mean(ecg ** 2)) + 1e-10

    # ── try loading real noise ────────────────────────────────────────────────
    try:
        bw_n = _load_noise_record("bw", N, fs, noise_dir, seed=seed)
        em_n = _load_noise_record("em", N, fs, noise_dir, seed=seed + 1)
        ma_n = _load_noise_record("ma", N, fs, noise_dir, seed=seed + 2)
    except FileNotFoundError:
        print("  ⚠  Real noise files not found in NOISE_DIR "
              f"({noise_dir}).\n"
              "     Falling back to synthetic noise.\n"
              "     Set NOISE_DIR in ecg_signals.py or pass noise_dir=...")
        return add_noise(ecg, fs=fs, snr_db=snr_db, seed=seed)

    w_bw, w_em, w_ma = weights
    mixed_noise = w_bw * bw_n + w_em * em_n + w_ma * ma_n

    # Scale mixed noise to target SNR
    noise_rms_target = sig_rms / (10 ** (snr_db / 20.0))
    mixed_rms = np.sqrt(np.mean(mixed_noise ** 2)) + 1e-10
    mixed_noise = mixed_noise * (noise_rms_target / mixed_rms)

    # Add synthetic 50 Hz hum (power-line interference)
    rng = np.random.default_rng(seed)
    hum = 0.25 * sig_rms * np.sin(2 * np.pi * 50.0 * t + rng.uniform(0, 2 * np.pi))

    return ecg + mixed_noise + hum


def add_noise(ecg, fs=500, snr_db=5, seed=99):
    """
    Add three layers of apocalypse noise ON TOP of the real ECG signal.

    The real MIT-BIH recordings already contain clinical noise (electrode
    motion, EMG artefact, etc). This adds extra interference to simulate
    the destroyed-infrastructure environment:

    Layer 1 — White Gaussian noise
        Amplitude set so the ADDED noise gives the target SNR relative
        to the ECG. At snr_db=5 the signal is very hard to see by eye.

    Layer 2 — Baseline wander at 0.2 Hz
        Slow sinusoidal drift, random phase, amplitude = 30% of RMS.
        Models patient movement or damaged electrode contact.

    Layer 3 — 50 Hz power-line hum
        Pure sine at 50 Hz, random phase, amplitude = 25% of RMS.
        The DFT notch filter removes this precisely — demonstrating
        a key advantage of frequency-domain over time-domain filtering.
    """
    rng     = np.random.default_rng(seed)
    N       = len(ecg)
    t       = np.arange(N) / fs
    sig_rms = np.sqrt(np.mean(ecg ** 2)) + 1e-10

    noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    white     = rng.normal(0, noise_rms, N)
    wander    = 0.30 * sig_rms * np.sin(2*np.pi*0.20*t + rng.uniform(0, 2*np.pi))
    hum       = 0.25 * sig_rms * np.sin(2*np.pi*50.0*t + rng.uniform(0, 2*np.pi))

    return ecg + white + wander + hum