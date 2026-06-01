## Project idea

Real ECG recordings are always contaminated by noise: breathing drift, electrode motion, muscle interference and hum. 
This project:
- demonstrates how the Discrete Fourier Transform (DFT) can be used to analyse and recover clean ECG signal - DFT converts the ECG    signal from the time domain into the frequency domain, which makes it possible to identify frequency components that belong to the heartbeat and components that are likely noise, 
- estimates heart rate by identifying the dominant frequency in the 0.5 – 3.5 Hz band of the DFT magnitude spectrum.

 A fictional “zombie apocalypse” scenario is introduced for entertainmnet purpose of the video. The “zombie” and “infected” labels are creative presentation labels. However, the underlying ECG recordings are real clinical ECG signals from the MIT-BIH Arrhythmia Database.

 # Note: 
 Since database ECG records do not contain any disruptive noises they are first added by us to be then filtered out by DFT. The noise signals are not generated or created by us. They were obtained from a publicly available real-world ECG noise database and incorporated into the project. 

## License / Data used 
MIT-BIH datasets belong to PhysioNet: https://physionet.org/

Three main records were chosen from MIT-BIH Arrhythmia Database to demonstarte distinct signals for healthy, arrhytmic (turning) and bradycardic (zombie) heart rates. 
- '100' — healthy, normal sinus rhythm 
- '203' — turning stage, irregular rhythm (arrhythmia)
- '119' — zombie, inverted (fictional aspect created as zombie characterictic, amplitude is multiplied by -1 )
          and reduced in amplitude which represents a weak "dying" heart (bradycardia)
          
The real noise components are mixed with the clean ECG and scaled to a chosen signal-to-noise ratio. 
They are from MIT-BIH Noise Stress Test Database:
- 'bw' — baseline wander
- 'em' — electrode motion artifact
- 'ma' — muscle artifact

## How to run

Requirements:
- Python 3.9 or newer
- packages: numpy matplotlib wfdb 
  instal by putting the following command in the terminal: pip install numpy matplotlib wfdb

1. Download this repository and make sure the 'mitdb/' and 'nstdb/' folders are inside the project folder.
2. Run run_me.py
3. The script prints BPM readings to the terminal and saves figures the 'figures/' folder:
   - 'fig1_main_noisy_vs_recovered.png' - Noisy vs DFT-recovered ECG for all three signal types
   - 'fig2_dft_filter_explanation.png' - DFT magnitude spectrum before and after filtering
   - 'fig3_stress_tests.png' - Window length sensitivity and noise robustness plots
   - 'fig4_dft_failure.png' - one DFT over whole signal vs repeated DFTs over short windows, showing non-stationarity 

## File explanations
# refer to respective files for code walkthrough

1. 'ecg_signals.py' - loads all data and adds noise.
2. 'dft_filter.py' - contains all DFT-based signal processing.
3. 'stress_tests.py':
   a. 'window_sensitivity()' - size of signal windows change the accuracy of BPM as DFT can only distinguish two frequencies if they're apart by at least certain frequency difference.
      - If window length is small, frequency differenc is large and the spectrum is blurry. 
      - If window length is large, frequency differenc is small and the spectrum is sharp. 
   b. 'noise_robustness()' - how much noise can DFT filter handle before it reports wrong heart rate.
   c. 'failure_cases()' - DFT assumes the signal's properties are constant over the entire window (stationary). Irregular signals (non-stationary) violate this as their beat timing is random and changes throughout the recording.
4. 'plots.py' - generates and saves all figures.
