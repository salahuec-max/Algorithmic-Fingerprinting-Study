# Algorithmic Fingerprinting in Computational Photography
**Longitudinal Study of Sub-Pixel Imaging Artifacts: iPhone vs. Samsung Ecosystems**

## Overview
This repository contains the digital signal processing (DSP) pipeline and analysis scripts used to identify and filter deterministic noise in quantum-based environmental sensors. 

### Key Methodology
* **Signal Capture:** 60s sampling @ 1000Hz (60,000 samples).
* **Spectral Analysis:** Fast Fourier Transform (FFT) with a Hanning window.
* **Noise Mitigation:** Suppression of 50Hz power-line coupling and mechanical harmonics.
* **Entropy Distillation:** Von Neumann unbiasing for cryptographic-quality randomness.

## Repository Structure
- `analysis.py`: Main Python script for FFT and signal characterization.
- `Dockerfile`: Container configuration for environment reproducibility.
- `raw_data.csv`: (Placeholder) Sample sensor data logs.

## Reproducibility
To run the analysis via Docker:
```bash
docker build -t quantum-noise-analysis .
docker run quantum-noise-analysis
