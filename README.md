# zombiept2
# Zombie Apocalypse ECG — DFT Signal Recovery Project

This project demonstrates how the Discrete Fourier Transform (DFT) can be used to analyse, corrupt, recover, and sonify ECG signals in a “zombie apocalypse” scenario.

The project uses real ECG recordings from the MIT-BIH Arrhythmia Database and applies DFT-based filtering to recover heart signals after noise corruption. It compares different heart types, including a normal heart, a weak/inverted bradycardia signal, and an irregular “zombie” arrhythmia signal.

## Project idea

The goal of the project is to show that the DFT is not only useful for visualising frequencies, but can also be used as an actual signal-processing tool.

The project:

* loads real ECG recordings,
* adds realistic physiological noise,
* removes unwanted frequency components using DFT filtering,
* estimates heart rate from the frequency spectrum,
* shows limitations of DFT analysis,
* generates plots and audio files for presentation/video use.

## Data used

The ECG signals are taken from the MIT-BIH Arrhythmia Database.

The main records used are:

* `100` — normal sinus rhythm, used as the healthy survivor baseline;
* `119` — bradycardia/conduction abnormality, inverted and reduced in amplitude to represent a weak “dying” heart;
* `203` — irregular arrhythmia, used as the chaotic “zombie” heart.

Additional records are loaded for comparison and failure-case analysis, including records such as `116`, `118`, `122`, `200`, and `207`.

## Noise used

The main recovery pipeline uses real noise recordings from the MIT-BIH Noise Stress Test Database.

The real noise components are:

* `bw` — baseline wander;
* `em` — electrode motion artifact;
* `ma` — muscle artifact.

These noise signals are mixed with the clean ECG and scaled to a chosen signal-to-noise ratio.

A synthetic 50 Hz power-line hum is also added to demonstrate how a DFT notch filter can remove a precise unwanted frequency.

If the real noise database files are not available, the code falls back to synthetic noise generated in Python. The separate noise robustness stress test also uses synthetic noise to test performance across different SNR levels.

## Repository structure

```text
zombiept2/
│
├── run_me.py
├── ecg_signals.py
├── dft_filter.py
├── plots.py
├── stress_tests.py
├── README.md
│
├── mitdb/
│   └── MIT-BIH Arrhythmia Database files
│
├── nstdb/
│   └── MIT-BIH Noise Stress Test Database files
│
└── figures/
    └── generated plots and audio files
```

## File explanations

### `run_me.py`

This is the main script. Run this file to generate all outputs.

It performs the full workflow:

1. loads the real ECG recordings;
2. estimates BPM using the DFT;
3. plots the three main heart types;
4. adds noise to the signals;
5. recovers the ECG using DFT notch and bandpass filtering;
6. saves recovery plots;
7. generates audio versions of the ECG signals;
8. runs stress tests;
9. saves all figures into the `figures/` folder.

Run with:

```bash
python run_me.py
```

### `ecg_signals.py`

This file loads and prepares the ECG signals.

It contains functions for:

* loading MIT-BIH records using `wfdb`;
* resampling signals to 500 Hz;
* removing DC offset;
* generating the normal, bradycardia, and arrhythmia cases;
* loading additional interesting MIT-BIH cases;
* adding real NSTDB noise;
* adding synthetic fallback noise.

Main functions include:

```python
generate_normal()
generate_bradycardia()
generate_arrhythmia()
load_all_interesting_cases()
add_real_noise()
add_noise()
```

### `dft_filter.py`

This file contains the DFT-based signal-processing methods.

It includes:

* DFT magnitude spectrum calculation;
* DFT bandpass filtering;
* DFT notch filtering;
* full ECG recovery;
* heart-rate estimation from the DFT spectrum.

Main functions include:

```python
spectral_magnitude()
dft_bandpass()
dft_notch()
recover_ecg()
heart_rate_dft()
```

The recovery pipeline first removes 50 Hz hum using a DFT notch filter, then keeps only the ECG frequency band using a DFT bandpass filter.

### `plots.py`

This file contains all plotting functions.

It saves figures into the `figures/` folder and uses a dark visual style to match the zombie-apocalypse theme.

It creates plots such as:

* comparison of the three heart types;
* clean/noisy/recovered ECG signals;
* DFT spectra before and after filtering;
* window sensitivity results;
* noise robustness results;
* DFT failure cases;
* additional interesting MIT-BIH records.

### `stress_tests.py`

This file tests how reliable the DFT method is under different conditions.

It includes:

1. **Window sensitivity**
   Tests different DFT window lengths and shows how frequency resolution affects BPM estimation.

2. **Noise robustness**
   Tests heart-rate estimation at different SNR levels using synthetic noise.

3. **Failure cases**
   Shows that a global DFT can be misleading for non-stationary or highly irregular ECG signals, such as the arrhythmia/zombie case.

## Requirements

Install the required Python packages:

```bash
pip install numpy matplotlib scipy wfdb
```

## How to run

1. Clone or download this repository.
2. Make sure the `mitdb/` folder is inside the project folder.
3. Make sure the `nstdb/` folder is inside the project folder if you want to use real noise.
4. Run:

```bash
python run_me.py
```

5. Check the `figures/` folder for generated plots and audio files.

## Expected outputs

The script generates figures such as:

```text
fig1_three_hearts.png
fig2_recovery_normal.png
fig2_recovery_bradycardia.png
fig2_recovery_arrhythmia.png
fig3a_window_spectra.png
fig3b_window_accuracy.png
fig4_noise_robustness.png
fig5_failure_cases.png
fig6_interesting_cases.png
```

It also creates audio files such as:

```text
audio_normal_original.wav
audio_normal_distorted.wav
audio_normal_recovered.wav
audio_bradycardia_original.wav
audio_bradycardia_distorted.wav
audio_bradycardia_recovered.wav
audio_arrhythmia_original.wav
audio_arrhythmia_distorted.wav
audio_arrhythmia_recovered.wav
```

## Method summary

The DFT converts the ECG signal from the time domain into the frequency domain. This makes it possible to identify which frequency components belong to the heartbeat and which components are likely noise.

The recovery process works as follows:

1. Compute the DFT of the noisy ECG.
2. Remove the 50 Hz power-line hum using a notch filter.
3. Keep only the main ECG frequency range using a bandpass filter.
4. Use the inverse DFT to reconstruct the cleaned ECG signal.
5. Estimate BPM from the fundamental frequency peak in the DFT spectrum.

## Notes

The project is intended as a Signals and Systems demonstration. It is not a medical diagnostic tool.

The “zombie” and “dying heart” labels are creative presentation labels. The underlying ECG recordings are real clinical ECG signals from the MIT-BIH Arrhythmia Database, but the story interpretation is fictional.
