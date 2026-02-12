Algorithmic Fingerprinting in Computational Photography
Longitudinal Study of Sub-Pixel Imaging Artifacts: iPhone vs. Samsung Ecosystems
Overview
This repository contains the digital signal processing (DSP) pipeline used to identify and quantify deterministic noise and algorithmic artifacts in mobile imaging sensors. Originally developed for high-precision characterization of quantum-based sensor noise, this framework has been adapted to isolate vendor-specific "algorithmic fingerprints" in the iPhone and Samsung Galaxy ecosystems.

Key Methodology
Spectral Analysis: Fast Fourier Transform (FFT) with a Hanning window to expose periodic grid artifacts (GAS).

Deterministic Noise Extraction: Identification of non-stochastic sub-pixel variations across 1423x2919 and 3000x3702 matrices.

Statistical Decomposition: Automated calculation of Kurtosis, Skewness, and the Computational Photography Artifacts Index (CPAI).

Forensic Modeling: Distinguishing between standard ISP output (iPhone Deep Fusion) and generative AI synthesis (Samsung S25 Ultra).

Repository Structure
analysis.py: Main Python script implementing the Fourier-based forensic pipeline.

Dockerfile: Container configuration for full environmental reproducibility (consistent with Scientific Reports standards).

raw_data.csv: Normalized statistical metadata extracted from the study's longitudinal image database.
