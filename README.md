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

import numpy as np
from PIL import Image
from scipy.fftpack import fft2, fftshift
from scipy.stats import kurtosis, skew
from scipy.ndimage import sobel, gaussian_filter
from skimage.feature import local_binary_pattern
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ForensicAnalyzer:
    def __init__(self, image_path, use_adaptive_windowing=True):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.img_array = np.array(self.img.convert('L'))
        self.use_adaptive_windowing = use_adaptive_windowing
        self.filename = image_path.split('/')[-1]
        
    def calculate_statistical_fingerprint(self):
        """Extracts higher-order moments from pixel distribution with robust outlier handling."""
        flat_pixels = self.img_array.flatten()
        # Clip extreme outliers (saturated pixels, lens flare) for robust statistics
        p_low, p_high = np.percentile(flat_pixels, [1, 99])
        flat_clipped = flat_pixels[(flat_pixels >= p_low) & (flat_pixels <= p_high)]
        
        return {
            'kurtosis': kurtosis(flat_clipped, fisher=True),  # Fisher=0 for normal distribution
            'skewness': skew(flat_clipped),
            'std_dev': np.std(flat_clipped),
            'entropy': self._calculate_entropy(flat_clipped),
            'dynamic_range': p_high - p_low
        }
    
    def _calculate_entropy(self, pixel_array):
        """Shannon entropy of pixel distribution."""
        hist, _ = np.histogram(pixel_array, bins=256, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def fourier_grid_detection(self):
        """
        Enhanced grid artifact detection with noise floor compensation.
        Returns multiple metrics for CFA pattern analysis.
        """
        # Apply windowing to reduce spectral leakage
        h, w = self.img_array.shape
        hanning_h = np.hanning(h)
        hanning_w = np.hanning(w)
        window = np.outer(hanning_h, hanning_w)
        img_windowed = self.img_array * window
        
        # Compute FFT
        f_transform = fft2(img_windowed)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        log_spectrum = 20 * np.log10(magnitude_spectrum + 1)
        
        # Adaptive region sizing based on image dimensions
        center_h, center_w = h // 2, w // 2
        region_size = min(100, min(h, w) // 5)
        
        # Define frequency bands
        dc_region = log_spectrum[center_h-5:center_h+5, center_w-5:center_w+5]
        
        # Mid-frequency (texture region)
        mf_region = log_spectrum[
            center_h-region_size:center_h+region_size,
            center_w-region_size:center_w+region_size
        ]
        
        # High-frequency (artifact region)
        hf_region = log_spectrum[
            center_h-region_size//2:center_h+region_size//2,
            center_w-region_size//2:center_w+region_size//2
        ]
        
        # Remove DC component influence
        dc_median = np.median(dc_region)
        mf_clean = mf_region - dc_median
        hf_clean = hf_region - dc_median
        
        # Calculate multiple artifact metrics
        grid_score = np.std(hf_clean) / (np.mean(np.abs(mf_clean)) + 1e-8)
        
        # Radial average of frequency spectrum (for periodic pattern detection)
        y, x = np.indices((hf_clean.shape))
        center = np.array([hf_clean.shape[0]//2, hf_clean.shape[1]//2])
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        radial_profile = np.bincount(r.ravel(), hf_clean.ravel()) / np.bincount(r.ravel())
        
        # Detect strong periodic peaks
        radial_peaks = np.diff(radial_profile)
        peak_strength = np.std(radial_peaks[5:20]) if len(radial_peaks) > 20 else 0
        
        return {
            'gas': grid_score,
            'peak_strength': peak_strength,
            'dc_median': dc_median,
            'hf_noise_floor': np.median(hf_clean)
        }, log_spectrum
    
    def get_spatial_asymmetry(self):
        """
        Comprehensive spatial heterogeneity analysis.
        Measures non-physical light distribution and demosaicing directionality.
        """
        h, w = self.img_array.shape
        
        # Quadrant analysis
        q1 = np.mean(self.img_array[:h//2, :w//2])
        q2 = np.mean(self.img_array[:h//2, w//2:])
        q3 = np.mean(self.img_array[h//2:, :w//2])
        q4 = np.mean(self.img_array[h//2:, w//2:])
        
        # Diagonal asymmetry (photographic lenses are radially symmetric)
        diag_asymmetry = abs(q1 - q4) + abs(q2 - q3)
        
        # Horizontal vs vertical gradient bias (demosaicing direction preference)
        grad_x = sobel(self.img_array, axis=1)
        grad_y = sobel(self.img_array, axis=0)
        direction_bias = (np.mean(np.abs(grad_x)) - np.mean(np.abs(grad_y))) / (np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y)) + 1e-8)
        
        # Center-edge falloff (vignetting signature)
        center_region = self.img_array[h//4:3*h//4, w//4:3*w//4]
        edge_region = np.concatenate([
            self.img_array[:h//8, :].flatten(),
            self.img_array[7*h//8:, :].flatten(),
            self.img_array[:, :w//8].flatten(),
            self.img_array[:, 7*w//8:].flatten()
        ])
        vignetting_ratio = np.mean(edge_region) / (np.mean(center_region) + 1e-8)
        
        asymmetry_index = diag_asymmetry / (np.mean(self.img_array) + 1e-8)
        
        return {
            'asymmetry_index': asymmetry_index,
            'direction_bias': direction_bias,
            'vignetting_ratio': vignetting_ratio,
            'quadrant_means': [q1, q2, q3, q4]
        }
    
    def extract_noise_fingerprint(self):
        """
        Photo-Response Non-Uniformity (PRNU) approximation.
        Device-specific sensor noise pattern.
        """
        # Denoise using Gaussian filter
        denoised = gaussian_filter(self.img_array.astype(float), sigma=1)
        noise_pattern = self.img_array.astype(float) - denoised
        
        # Statistical properties of noise
        noise_flat = noise_pattern.flatten()
        noise_std = np.std(noise_flat)
        noise_skew = skew(noise_flat)
        noise_kurtosis = kurtosis(noise_flat, fisher=True)
        
        # Frequency characteristics of noise
        f_noise = fft2(noise_pattern)
        f_noise_shift = fftshift(f_noise)
        noise_spectrum = np.abs(f_noise_shift)
        
        # Blue noise vs pink noise signature
        center = [dim // 2 for dim in noise_pattern.shape]
        low_freq_noise = np.mean(noise_spectrum[center[0]-5:center[0]+5, center[1]-5:center[1]+5])
        high_freq_noise = np.mean(noise_spectrum[center[0]-50:center[0]+50, center[1]-50:center[1]+50]) - low_freq_noise
        
        return {
            'noise_std': noise_std,
            'noise_skew': noise_skew,
            'noise_kurtosis': noise_kurtosis,
            'noise_high_low_ratio': high_freq_noise / (low_freq_noise + 1e-8),
            'noise_pattern': noise_pattern[:64, :64]  # Store thumbnail for comparison
        }
    
    def texture_analysis(self):
        """
        LBP-based texture fingerprint for AI vs optical differentiation.
        """
        # Local Binary Pattern for microtexture analysis
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(self.img_array, n_points, radius, method='uniform')
        
        # LBP histogram as texture signature
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # Texture uniformity
        texture_entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        texture_uniformity = np.sum(hist**2)
        
        return {
            'lbp_histogram': hist,
            'texture_entropy': texture_entropy,
            'texture_uniformity': texture_uniformity,
            'dominant_pattern': np.argmax(hist)
        }
    
    def full_forensic_report(self):
        """Comprehensive fingerprint extraction."""
        stats = self.calculate_statistical_fingerprint()
        fft_metrics, spectrum = self.fourier_grid_detection()
        asymmetry_metrics = self.get_spatial_asymmetry()
        noise_fingerprint = self.extract_noise_fingerprint()
        texture = self.texture_analysis()
        
        report = {
            'filename': self.filename,
            'statistical': stats,
            'fft_artifacts': fft_metrics,
            'spatial': asymmetry_metrics,
            'noise': noise_fingerprint,
            'texture': texture
        }
        
        return report
    
    def print_summary(self, report=None):
        """Formatted output for quick forensic assessment."""
        if report is None:
            report = self.full_forensic_report()
            
        print(f"\n{'='*50}")
        print(f"🔬 FORENSIC ANALYSIS REPORT: {report['filename']}")
        print(f"{'='*50}")
        
        print(f"\n📊 STATISTICAL FINGERPRINT:")
        print(f"  • Kurtosis (Texture Sharpness):  {report['statistical']['kurtosis']:.4f}")
        print(f"  • Skewness:                      {report['statistical']['skewness']:.4f}")
        print(f"  • Entropy (Information Density): {report['statistical']['entropy']:.4f}")
        print(f"  • Dynamic Range:                 {report['statistical']['dynamic_range']:.1f} levels")
        
        print(f"\n🔄 FREQUENCY DOMAIN ARTIFACTS:")
        print(f"  • Grid Artifact Score (GAS):     {report['fft_artifacts']['gas']:.4f}")
        print(f"  • Periodic Peak Strength:        {report['fft_artifacts']['peak_strength']:.4f}")
        print(f"  • HF Noise Floor:                {report['fft_artifacts']['hf_noise_floor']:.4f}")
        
        print(f"\n🎛️ SPATIAL HETEROGENEITY:")
        print(f"  • Asymmetry Index:               {report['spatial']['asymmetry_index']:.4f}")
        print(f"  • Demosaicing Direction Bias:    {report['spatial']['direction_bias']:+.4f}")
        print(f"  • Vignetting Ratio (Edge/Center):{report['spatial']['vignetting_ratio']:.4f}")
        
        print(f"\n📷 SENSOR NOISE FINGERPRINT:")
        print(f"  • Noise σ:                       {report['noise']['noise_std']:.4f}")
        print(f"  • Noise Skew:                    {report['noise']['noise_skew']:.4f}")
        print(f"  • High/Low Freq Noise Ratio:     {report['noise']['noise_high_low_ratio']:.4f}")
        
        print(f"\n🧩 TEXTURE SIGNATURE:")
        print(f"  • Texture Entropy:               {report['texture']['texture_entropy']:.4f}")
        print(f"  • Texture Uniformity:            {report['texture']['texture_uniformity']:.4f}")
        print(f"  • Dominant LBP Pattern:          {report['texture']['dominant_pattern']}")
        
        return report

if __name__ == "__main__":
    import os
    import sys
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_filename = sys.argv[1]
    else:
        image_filename = 'image.jpg'
    
    # Check if file exists
    if not os.path.exists(image_filename):
        print(f"⚠️ {image_filename} not found. Please provide a valid image path.")
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)
    
    # Run comprehensive forensic analysis
    try:
        analyzer = ForensicAnalyzer(image_filename)
        report = analyzer.print_summary()
        
        # Optional: Export to CSV/JSON for longitudinal study
        # pd.DataFrame([report]).to_csv(f"{image_filename}_forensic.csv")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
Reproducibility
To run the analysis via Docker:

docker build -t forensic-analysis.

docker run forensic-analysis.
