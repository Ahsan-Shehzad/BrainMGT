# Multi_band_Temporal_Scales.py

import numpy as np
import pywt
from scipy.signal import butter, filtfilt
import os

class MultiBandTemporalScales:
    def __init__(self, preprocessed_fmri_data, hierarchical_scales, output_dir):
        self.fmri_data = preprocessed_fmri_data
        self.hierarchical_scales = hierarchical_scales
        self.output_dir = output_dir

    def _bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Helper function to apply a Butterworth bandpass filter.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data

    def _dwt_filter(self, data, wavelet, mode='zero'):
        """
        Helper function to apply Discrete Wavelet Transform (DWT) filtering.
        """
        coeffs = pywt.wavedec(data, wavelet, mode=mode)
        return coeffs

    def fast_frequency_band(self, microscale_data, fs=1.0):
        """
        Extract fast frequency band signals (0.08-0.25 Hz) using DWT on microscale data.
        Justification: This range captures faster neural dynamics, linked to transient brain processes.
        """
        # Using a Butterworth filter as an additional step for bandpass filtering within the specified range.
        lowcut = 0.08
        highcut = 0.25
        filtered_data = self._bandpass_filter(microscale_data, lowcut, highcut, fs)

        # Apply DWT for more frequency decomposition, using a suitable wavelet (e.g., 'db4').
        wavelet = 'db4'
        dwt_coeffs = self._dwt_filter(filtered_data, wavelet)

        np.save(os.path.join(self.output_dir, 'fast_frequency_band.npy'), dwt_coeffs)
        return dwt_coeffs

    def intermediate_frequency_band(self, mesoscale_data, fs=1.0):
        """
        Extract intermediate frequency band signals (0.027-0.073 Hz) using DWT on mesoscale data.
        Justification: This range is associated with slower cognitive processes and intermediate neural dynamics.
        """
        lowcut = 0.027
        highcut = 0.073
        filtered_data = self._bandpass_filter(mesoscale_data, lowcut, highcut, fs)

        wavelet = 'db4'
        dwt_coeffs = self._dwt_filter(filtered_data, wavelet)

        np.save(os.path.join(self.output_dir, 'intermediate_frequency_band.npy'), dwt_coeffs)
        return dwt_coeffs

    def slow_frequency_band(self, macroscale_data, fs=1.0):
        """
        Extract slow frequency band signals (0.01-0.027 Hz) using DWT on macroscale data.
        Justification: Low-frequency dynamics are associated with long-range brain connectivity and resting-state activity.
        """
        lowcut = 0.01
        highcut = 0.027
        filtered_data = self._bandpass_filter(macroscale_data, lowcut, highcut, fs)

        wavelet = 'db4'
        dwt_coeffs = self._dwt_filter(filtered_data, wavelet)

        np.save(os.path.join(self.output_dir, 'slow_frequency_band.npy'), dwt_coeffs)
        return dwt_coeffs

    def run_temporal_scales(self):
        """
        Execute the full pipeline for extracting multi-band temporal scales from preprocessed fMRI data and hierarchical spatial scales.
        """
        microscale_data = self.hierarchical_scales['microscale']
        mesoscale_data = self.hierarchical_scales['mesoscale']
        macroscale_data = self.hierarchical_scales['macroscale']

        # Extract temporal scales
        fast_band = self.fast_frequency_band(microscale_data)
        intermediate_band = self.intermediate_frequency_band(mesoscale_data)
        slow_band = self.slow_frequency_band(macroscale_data)

        return fast_band, intermediate_band, slow_band

# Example usage:
# hierarchical_scales = {
#     'microscale': np.load('path_to_microscale_data.npy'),
#     'mesoscale': np.load('path_to_mesoscale_data.npy'),
#     'macroscale': np.load('path_to_macroscale_data.npy')
# }
# temporal_scales = MultiBandTemporalScales(preprocessed_fmri_data="path_to_preprocessed_fmri_data.nii.gz", hierarchical_scales=hierarchical_scales, output_dir="output_directory")
# temporal_scales.run_temporal_scales()
