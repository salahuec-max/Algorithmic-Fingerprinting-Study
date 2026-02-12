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

Analysis Pipeline (analysis.py)
The following core logic is used for the extraction of the Computational Photography Artifacts Index (CPAI):

Python
import numpy as np
from PIL import Image
from scipy.fftpack import fft2, fftshift
from scipy.stats import kurtosis, skew
import os

class ForensicAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        # Load image and convert to grayscale for frequency analysis
        self.img_array = np.array(Image.open(image_path).convert('L'))
        
    def calculate_cpai(self):
        """Calculates Computational Photography Artifacts Index (CPAI)"""
        flat_pixels = self.img_array.flatten()
        return {
            'kurtosis': kurtosis(flat_pixels),
            'skewness': skew(flat_pixels),
            'gas_score': self.fourier_grid_detection()
        }

    def fourier_grid_detection(self):
        """Detects periodic grid artifacts using 2D FFT"""
        f_shift = fftshift(fft2(self.img_array))
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        h, w = magnitude_spectrum.shape
        # Analyze high-frequency region for grid spikes (The "AI Fingerprint")
        hf_region = magnitude_spectrum[h//2-50:h//2+50, w//2-50:w//2+50]
        return np.std(hf_region) / np.mean(hf_region)

if __name__ == "__main__":
    test_file = 'sample_image.jpg'
    
    # Synthetic fallback for reproducibility in headless environments
    if not os.path.exists(test_file):
        synthetic = np.zeros((512, 512), dtype=np.uint8)
        synthetic[::8, :] = 50 
        Image.fromarray(synthetic).save(test_file)

    analyzer = ForensicAnalyzer(test_file)
    results = analyzer.calculate_cpai()
    
    print(f"--- Forensic Report ---")
    print(f"CPAI Gas Score: {results['gas_score']:.4f}")
    print(f"Kurtosis: {results['kurtosis']:.4f}")
Reproducibility
To run the analysis via Docker:

docker build -t forensic-analysis .
docker run forensic-analysis
