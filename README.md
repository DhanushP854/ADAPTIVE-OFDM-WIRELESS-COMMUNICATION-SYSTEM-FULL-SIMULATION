# ADAPTIVE OFDM WIRELESS COMMUNICATION SYSTEM – FULL SIMULATION

**A Complete End-to-End Simulation of an Adaptive OFDM Wireless Communication System Using Python**

**Author:** Dhanush P
**Project Domain:** Digital Wireless Communication, OFDM, Adaptive Modulation, and Signal Processing

---

**PROJECT OVERVIEW**

This project implements a complete end-to-end simulation of an Adaptive Orthogonal Frequency Division Multiplexing (OFDM) wireless communication system using Python.

The system transmits a text message through a simulated wireless channel affected by frequency-selective multipath Rayleigh fading and Additive White Gaussian Noise (AWGN). At the receiver, channel estimation, MMSE equalization, demodulation, and optional error correction are performed to recover the original message.

The project also evaluates system performance using Bit Error Rate (BER), Symbol Error Rate (SER), Error Vector Magnitude (EVM), throughput, and spectral efficiency under different Signal-to-Noise Ratio (SNR) conditions.

**OBJECTIVES**

- To implement a complete OFDM transmitter and receiver.
- To demonstrate adaptive modulation based on operating SNR.
- To simulate realistic wireless channel impairments.
- To estimate the wireless channel using pilot symbols and Least Squares (LS) estimation.
- To improve received signal quality using MMSE equalization.
- To implement optional Forward Error Correction (FEC) using repetition coding and block interleaving.
- To evaluate communication performance across different modulation schemes and SNR values.
- To visualize constellation diagrams, channel response, error rates, throughput, and spectral efficiency.

**KEY FEATURES**

- End-to-end text-to-text wireless communication simulation.
- Adaptive modulation using BPSK, QPSK, 16-QAM, and 64-QAM.
- OFDM framing with pilot insertion, IFFT/FFT processing, and cyclic prefix.
- Frequency-selective multipath Rayleigh fading channel.
- AWGN noise model with configurable SNR.
- Pilot-based LS channel estimation with linear interpolation.
- Minimum Mean Square Error (MMSE) frequency-domain equalization.
- Optional repetition-code FEC and block interleaving.
- BER, SER, and EVM performance analysis.
- Throughput and spectral efficiency calculations.
- Monte Carlo performance evaluation across multiple SNR values.
- Automatic saving of simulation plots as PNG images.
- Support for interactive execution in Google Colab/Jupyter and command-line execution.

**SYSTEM BLOCK DIAGRAM**

```text
                  TRANSMITTER
                       |
                 Input Text Message
                       |
                  Text to Bits
                       |
                Length Header Addition
                       |
              Optional FEC and Interleaving
                       |
                Adaptive Modulation
             BPSK / QPSK / 16-QAM / 64-QAM
                       |
                 OFDM Frame Builder
                       |
                  Pilot Insertion
                       |
                     IFFT
                       |
                 Cyclic Prefix
                       |
                       v
             WIRELESS CHANNEL
                       |
           Multipath Rayleigh Fading
                       |
                 AWGN Noise
                       |
                       v
                    RECEIVER
                       |
                 Cyclic Prefix Removal
                       |
                      FFT
                       |
             LS Channel Estimation
                       |
               Linear Interpolation
                       |
                MMSE Equalization
                       |
                    Demodulation
                       |
             Optional FEC Decoding
                and Deinterleaving
                       |
                  Bits to Text
                       |
                Recovered Message
                       |
                Performance Analysis
```

**TECHNOLOGIES USED**

- **Programming Language:** Python 3
- **Numerical Processing:** NumPy
- **Plotting and Visualization:** Matplotlib
- **Execution Environment:** Google Colab, Jupyter Notebook, or a Python terminal
- **Communication Techniques:** OFDM, digital modulation, pilot-based channel estimation, and equalization

**SUPPORTED MODULATION SCHEMES**

| Modulation | Bits per Symbol | Main Characteristic                        |
| ---------- | --------------: | ------------------------------------------ |
| BPSK       |               1 | Higher robustness at lower SNR             |
| QPSK       |               2 | Balance between robustness and data rate   |
| 16-QAM     |               4 | Higher data rate at suitable SNR           |
| 64-QAM     |               6 | Higher spectral efficiency at suitable SNR |

**ADAPTIVE MODULATION STRATEGY**

The system selects the modulation scheme according to predefined SNR thresholds.

| Operating SNR        | Selected Modulation |
| -------------------- | ------------------- |
| Below 5 dB           | BPSK                |
| 5 dB to below 12 dB  | QPSK                |
| 12 dB to below 20 dB | 16-QAM              |
| 20 dB and above      | 64-QAM              |

Higher-order modulation can increase data throughput and spectral efficiency, but generally requires better channel conditions to maintain reliable communication.

**OFDM SYSTEM CONFIGURATION**

The default simulation uses the following parameters:

| Parameter                           | Default Value                |
| ----------------------------------- | ---------------------------- |
| FFT size                            | 64                           |
| Cyclic prefix length                | 16 samples                   |
| Pilot spacing                       | 4 subcarriers                |
| Pilot subcarriers                   | 16                           |
| Data subcarriers                    | 48                           |
| Sampling frequency                  | 20 MHz                       |
| Channel model                       | Multipath Rayleigh fading    |
| Default number of channel taps      | 6                            |
| Noise model                         | AWGN                         |
| Channel estimation                  | LS with linear interpolation |
| Equalization                        | MMSE                         |
| Default live-demo modulation        | Adaptive                     |
| Default live-demo SNR               | 18 dB                        |
| Default live-demo repetition factor | 5                            |

These are simulation parameters and can be modified in the source code where applicable.

**FORWARD ERROR CORRECTION**

The live text-transmission demonstration optionally uses a repetition code with majority-vote decoding and block interleaving.

The repetition factor is configurable and must be a positive odd integer. The default value is 5.

The repetition code introduces redundancy to improve the probability of recovering transmitted bits correctly. Interleaving distributes repeated bits across the transmitted sequence to help reduce the effect of localized errors.

The BER, SER, and EVM Monte Carlo sweep uses uncoded transmission to compare the raw performance of the different modulation schemes.

**PERFORMANCE METRICS**

- **Bit Error Rate (BER):** Measures the proportion of incorrectly received bits.
- **Symbol Error Rate (SER):** Measures the proportion of incorrectly received modulation symbols.
- **Error Vector Magnitude (EVM):** Measures the RMS difference between transmitted and received constellation symbols, expressed as a percentage.
- **Throughput:** Measures useful payload data transmitted per second, expressed in bits per second or Mbps.
- **Spectral Efficiency:** Measures useful throughput per unit bandwidth, expressed in bits/s/Hz.
- **Text Recovery:** Compares the original text message with the recovered message.
- **Channel Estimation:** Compares the true and estimated channel frequency responses.

**SIMULATION OUTPUTS**

The program saves the following plots in the configured output directory, which defaults to `ofdm_outputs/`.

| Output File              | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `constellation.png`      | Transmitted and received constellation diagrams          |
| `channel_response.png`   | True and estimated channel magnitude and phase responses |
| `adaptive_selection.png` | Selected modulation scheme versus SNR                    |
| `throughput_se.png`      | Throughput and spectral efficiency versus SNR            |
| `ber_ser_evm_sweep.png`  | BER, SER, and EVM performance across SNR values          |

The BER/SER/EVM sweep plot is generated when the Monte Carlo sweep is enabled.

**INSTALLATION AND REQUIREMENTS**

Install Python 3 and the required Python libraries.

```bash
pip install numpy matplotlib
```

No additional communication-system library is required by the supplied implementation.

**HOW TO RUN THE PROJECT**

**Option 1: Google Colab or Jupyter Notebook**

1. Open the supplied simulation notebook in Google Colab or Jupyter Notebook.
2. Execute the cells in sequence.
3. Enter a text message, operating SNR, and modulation choice if prompted.
4. Allow the simulation and performance analysis to complete.
5. View the generated plots and transmission report.

**Option 2: Run as a Python Script**

Save the source code as `adaptive_ofdm.py` and run:

```bash
python3 adaptive_ofdm.py
```

The program prompts for the input text message, channel SNR, and modulation scheme. Press Enter to use the default values.

**CUSTOM EXECUTION EXAMPLES**

Run with a custom message, SNR, and adaptive modulation:

```bash
python3 adaptive_ofdm.py --text "Hello Wireless World" --snr 18 --mod ADAPTIVE
```

Run using QPSK:

```bash
python3 adaptive_ofdm.py --text "OFDM Communication" --snr 15 --mod QPSK
```

Run using BPSK:

```bash
python3 adaptive_ofdm.py --text "Wireless Simulation" --snr 8 --mod BPSK
```

Specify a custom output directory:

```bash
python3 adaptive_ofdm.py --outdir results
```

Run the simulation without the Monte Carlo sweep to reduce execution time:

```bash
python3 adaptive_ofdm.py --skip-sweep
```

Disable repetition-code FEC for the live text demonstration:

```bash
python3 adaptive_ofdm.py --no-fec
```

Change the repetition factor:

```bash
python3 adaptive_ofdm.py --rep 7
```

Adjust the number of Monte Carlo trials per SNR point:

```bash
python3 adaptive_ofdm.py --trials 20
```

**COMMAND-LINE OPTIONS**

| Option         | Description                                  |
| -------------- | -------------------------------------------- |
| `--text`       | Text message to transmit                     |
| `--snr`        | Operating SNR in dB                          |
| `--mod`        | Select BPSK, QPSK, 16QAM, 64QAM, or ADAPTIVE |
| `--outdir`     | Directory for saved simulation plots         |
| `--trials`     | Number of Monte Carlo trials per SNR point   |
| `--skip-sweep` | Skip the BER/SER/EVM sweep                   |
| `--no-fec`     | Disable FEC for the live demonstration       |
| `--rep`        | Set the repetition-code factor               |

**PROJECT WORKFLOW**

1. Convert the input text into a binary bitstream.
2. Add a 32-bit payload-length header.
3. Optionally apply repetition coding and block interleaving.
4. Select a modulation scheme based on the configured mode and SNR.
5. Map bits into digital modulation symbols.
6. Insert data and pilot symbols into the OFDM frequency-domain grid.
7. Apply IFFT processing and add the cyclic prefix.
8. Transmit the signal through the simulated multipath Rayleigh fading channel.
9. Add AWGN according to the configured SNR.
10. Remove the cyclic prefix and apply FFT processing.
11. Estimate the channel using LS pilot estimates and linear interpolation.
12. Apply MMSE equalization and demodulate the received symbols.
13. Optionally deinterleave and decode the received bits.
14. Recover the original text using the payload-length header.
15. Calculate performance metrics, display the transmission report, and save the plots.

**EXPECTED OBSERVATIONS**

- BPSK generally provides better robustness than higher-order modulation under challenging channel conditions.
- Higher-order modulation can improve data rate and spectral efficiency when SNR is sufficiently high.
- Multipath fading and AWGN can increase BER, SER, and EVM.
- Channel estimation and MMSE equalization help compensate for channel distortion.
- Repetition coding and interleaving can improve message recovery, although they cannot guarantee error-free transmission.
- Adaptive modulation changes the selected modulation scheme as the operating SNR changes.
- The simulation makes it possible to compare modulation schemes and study the trade-off between reliability and spectral efficiency.

Actual numerical results depend on the selected message, SNR, channel realization, random seed, modulation scheme, and simulation settings. Successful recovery is not guaranteed under all channel conditions.

**PROJECT APPLICATIONS**

- Academic study of OFDM wireless communication.
- M.Tech communication engineering projects and demonstrations.
- Digital modulation and demodulation experiments.
- Wireless channel estimation and equalization studies.
- BER and SER performance evaluation.
- Adaptive modulation research and simulation.
- Investigation of throughput and spectral efficiency trade-offs.

**LIMITATIONS**

- This is a software simulation, not a physical RF transmitter or receiver.
- The channel and noise are modeled mathematically.
- Adaptive modulation uses predefined SNR thresholds rather than a practical feedback-control protocol.
- The channel estimator averages pilot estimates across the frame under a quasi-static channel assumption.
- The lightweight repetition code is educational and is not a substitute for advanced channel coding.
- Performance curves are simulation results and should not be interpreted as measurements from real wireless hardware.

**FUTURE ENHANCEMENTS**

- Implement convolutional, LDPC, or Polar channel coding.
- Add channel coding schemes with soft-decision decoding.
- Investigate improved channel-estimation algorithms.
- Add Doppler effects and time-varying fading.
- Compare Zero-Forcing and MMSE equalization.
- Extend the model to MIMO-OFDM.
- Investigate practical adaptive-modulation feedback and target-BER-based switching.
- Evaluate performance under additional wireless channel models.

**CONCLUSION**

The Adaptive OFDM Wireless Communication System demonstrates an end-to-end digital wireless communication link using Python. It integrates adaptive modulation, OFDM processing, multipath Rayleigh fading, AWGN, pilot-based channel estimation, MMSE equalization, optional error correction, and performance visualization.

The simulation provides a practical foundation for understanding how modulation order, channel conditions, error correction, and receiver processing influence communication reliability and spectral efficiency.

**OUTPUT RECEIVED**

The simulation is designed to produce a transmission report containing the selected modulation scheme, operating SNR, transmitted and recovered text, text-match status, BER, SER, EVM, throughput, and spectral efficiency, along with the configured performance plots.

**PROJECT COMPLETED** ✅
