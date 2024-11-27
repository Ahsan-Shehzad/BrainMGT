# fMRI_Preprocessing.py

import os
import nibabel as nib
from nipype.interfaces import fsl, afni
from nilearn.image import smooth_img, resample_to_img
from nilearn.datasets import load_mni152_template
import numpy as np

class fMRI_Preprocessing:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir
        self.raw_img = nib.load(input_file)

    def motion_correction(self, method='MCFLIRT'):
        """
        Perform motion correction using FSL's MCFLIRT or AFNI's 3dvolreg.
        """
        if method == 'MCFLIRT':
            mcflt = fsl.MCFLIRT(in_file=self.input_file, out_file=os.path.join(self.output_dir, 'mc_corrected.nii.gz'))
            mcflt.run()
        elif method == '3dvolreg':
            volreg = afni.Volreg(in_file=self.input_file, out_file=os.path.join(self.output_dir, 'volreg_corrected.nii.gz'))
            volreg.run()
        else:
            raise ValueError("Unsupported method for motion correction. Use 'MCFLIRT' or '3dvolreg'.")

    def slice_timing_correction(self, method='AFNI'):
        """
        Perform slice timing correction.
        """
        if method == 'AFNI':
            slicetimer = afni.TShift(in_file=self.input_file, out_file=os.path.join(self.output_dir, 'slice_timing_corrected.nii.gz'))
            slicetimer.run()
        else:
            raise ValueError("Unsupported method for slice timing correction. Use 'AFNI'.")

    def spatial_normalization(self, target_template='MNI152'):
        """
        Normalize fMRI images to a common template such as MNI space.
        """
        if target_template == 'MNI152':
            template_img = load_mni152_template()
            normalized_img = resample_to_img(self.raw_img, template_img)
            nib.save(normalized_img, os.path.join(self.output_dir, 'normalized.nii.gz'))
        else:
            raise ValueError("Unsupported template for spatial normalization. Use 'MNI152'.")

    def smoothing(self, fwhm=6):
        """
        Apply Gaussian smoothing to the fMRI data.
        """
        smoothed_img = smooth_img(self.raw_img, fwhm=fwhm)
        nib.save(smoothed_img, os.path.join(self.output_dir, 'smoothed.nii.gz'))

    def noise_removal(self, method='CompCor'):
        """
        Perform noise removal using methods like CompCor or ICA.
        """
        if method == 'CompCor':
            # This is a placeholder for actual CompCor implementation
            # Typically, you'd extract noise components and regress them out
            print("CompCor noise removal not fully implemented. This would involve PCA and regression.")
        elif method == 'ICA':
            # Placeholder for ICA-based denoising
            print("ICA noise removal not fully implemented. This would involve independent component analysis and artifact rejection.")
        else:
            raise ValueError("Unsupported method for noise removal. Use 'CompCor' or 'ICA'.")

    def run_preprocessing(self):
        """
        Run the full preprocessing pipeline.
        """
        self.motion_correction()
        self.slice_timing_correction()
        self.spatial_normalization()
        self.smoothing()
        self.noise_removal()

# Example usage:
# fmri_preproc = fMRI_Preprocessing(input_file="path_to_fmri_data.nii.gz", output_dir="output_directory")
# fmri_preproc.run_preprocessing()
