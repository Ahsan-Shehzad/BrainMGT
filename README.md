# BrainMGT: Multi-scale Brain Graph Transformer for Brain Disease Diagnosis

## Introduction

**BrainMGT** is a novel transformer-based model designed to diagnose brain diseases by leveraging hierarchical multi-scale brain networks. This project captures brain connectivity patterns at different spatial (microscale, mesoscale, macroscale) and temporal (fast, intermediate, slow frequency bands) scales, and integrates them using a multi-scale brain graph transformer. The model takes in preprocessed fMRI data, generates hierarchical brain networks, and uses these networks to accurately classify brain disorders.

## Table of Contents

- [Introduction](#introduction)
- [How to Use the Code](#how-to-use-the-code)
  - [1. fMRI_Preprocessing.py](#1-fmri_preprocessingpy)
  - [2. Hierarchical_Spatial_Scales.py](#2-hierarchical_spatial_scalespy)
  - [3. Multi_band_Temporal_Scaless.py](#3-multi_band_temporal_scalesspy)
  - [4. Hierarchical_Multiscale_Connectivity.py](#4-hierarchical_multiscale_connectivitypy)
  - [5. Multi_scale_Brain_Graph_Transformers.py](#5-multi_scale_brain_graph_transformerspy)
  - [6. train_brainmgt.py](#6-train_brainmgtpy)
  - [7. test_brainmgt.py](#7-test_brainmgtpy)
- [Dependencies](#dependencies)
- [Citation](#Citation)

## How to Use the Code

### 1. fMRI_Preprocessing.py

**Objective:** Prepare fMRI data for analysis by correcting artifacts and standardizing the data.

- **Motion Correction:** Adjust the data for any motion artifacts (e.g., head movements).
- **Slice Timing Correction:** Align the acquisition times of all slices in the dataset.
- **Spatial Normalization:** Normalize the fMRI images to a common brain template (e.g., MNI space).
- **Smoothing:** Apply a Gaussian kernel to smooth the fMRI data and increase signal-to-noise ratio.
- **Noise Removal:** Regress out noise components using methods like CompCor or ICA.

To use:
```python
from fMRI_Preprocessing import fMRI_Preprocessing

preprocessing = fMRI_Preprocessing(input_file="path_to_fmri_data.nii.gz", output_dir="output_directory")
preprocessing.run_preprocessing()
```

### 2. Hierarchical_Spatial_Scales.py

**Objective:** Define brain networks at different spatial resolutions to capture the hierarchical organization of the brain.

- **Microscale:** Define 1000 nodes using the Schaefer atlas and extract fine-grained brain network details.
- **Mesoscale:** Detect intermediate-level brain communities (~500) using community detection algorithms (e.g., Louvain).
- **Macroscale:** Aggregate mesoscale communities to define larger brain network organization (~100 communities).

To use:
```python
from Hierarchical_Spatial_Scales import HierarchicalSpatialScales

spatial_scales = HierarchicalSpatialScales(preprocessed_fmri="path_to_preprocessed_fmri.nii.gz", output_dir="output_directory")
spatial_scales.run_hierarchical_scales()
```

### 3. Multi_band_Temporal_Scaless.py

**Objective:** Analyze brain activity at different temporal resolutions (fast, intermediate, slow frequency bands) to capture dynamic changes across frequencies.

- **Fast Frequency Band (0.08-0.25 Hz):** Isolate high-frequency brain activity.
- **Intermediate Frequency Band (0.027-0.073 Hz):** Capture mid-frequency brain activity.
- **Slow Frequency Band (0.01-0.027 Hz):** Identify low-frequency brain activity.

To use:
```python
from Multi_band_Temporal_Scaless import MultiBandTemporalScales

temporal_scales = MultiBandTemporalScales(preprocessed_fmri_data="path_to_preprocessed_fmri.nii.gz", hierarchical_scales={
    'microscale': np.load("microscale_path.npy"),
    'mesoscale': np.load("mesoscale_path.npy"),
    'macroscale': np.load("macroscale_path.npy")
}, output_dir="output_directory")
temporal_scales.run_temporal_scales()
```

### 4. Hierarchical_Multiscale_Connectivity.py

**Objective:** Construct brain networks for each spatial and temporal scale to capture detailed connectivity patterns.

- **Intra-module Connectivity Estimation:** Measure connectivity within each module at different scales using Pearson correlation.
- **Global Adjacency Matrix Construction:** Integrate module-specific connectivity matrices into a comprehensive global network for each scale.
- **Inter-module Connectivity Estimation:** Measure connectivity between different modules within each scale.
- **Differential Thresholding:** Enhance intra-module connectivity while reducing inter-module connectivity to reflect hierarchical brain organization.
- **Multi-scale Brain Network Dataset:** Process fMRI datasets to generate a multiscale brain network dataset.

To use:
```python
from Hierarchical_Multiscale_Connectivity import HierarchicalMultiscaleConnectivity

connectivity = HierarchicalMultiscaleConnectivity(
    preprocessed_fmri="path_to_preprocessed_fmri.nii.gz",
    hierarchical_scales={'microscale': np.load("microscale.npy"),
                         'mesoscale': np.load("mesoscale.npy"),
                         'macroscale': np.load("macroscale.npy")},
    temporal_scales={'fast': np.load("fast_frequency.npy"),
                     'intermediate': np.load("intermediate_frequency.npy"),
                     'slow': np.load("slow_frequency.npy")},
    output_dir="output_directory"
)
connectivity.generate_multiscale_network_dataset()
```

### 5. Multi_scale_Brain_Graph_Transformers.py

**Objective:** Develop a transformer-based model that leverages hierarchical multi-scale brain networks to accurately diagnose brain diseases.

- **Input Embeddings:** Use Graph Attention Networks (GATs) to generate node and edge embeddings at each scale.
- **Positional Encoding:** Use learnable positional embeddings for nodes in the graph.
- **Scale-specific Encoders:** Capture dependencies and interactions within each scale using self-attention layers.
- **Cross-scale Encoders:** Capture interactions between scales using cross-attention mechanisms.
- **Hierarchical Adaptive Fusion:** Adaptively integrate features from scale-specific and cross-scale encoders.
- **Classification:** Use fully connected layers with softmax activation to classify brain disorders.

To use:
```python
from Multi_scale_Brain_Graph_Transformers import MultiScaleBrainGraphTransformer

model = MultiScaleBrainGraphTransformer(num_nodes=1000, num_heads=8, hidden_dim=128, output_dim=128, num_classes=2)
```

### 6. train_brainmgt.py

**Objective:** Train the BrainMGT model on a multi-scale brain network dataset.

- **Preprocess the fMRI data using `fMRI_Preprocessing.py`.**
- **Generate hierarchical spatial scales using `Hierarchical_Spatial_Scales.py`.**
- **Generate multi-band temporal scales using `Multi_band_Temporal_Scaless.py`.**
- **Generate hierarchical multiscale connectivity using `Hierarchical_Multiscale_Connectivity.py`.**
- **Train the BrainMGT model using `Multi_scale_Brain_Graph_Transformers.py`.**

To use:
```bash
python train_brainmgt.py
```

### 7. test_brainmgt.py

**Objective:** Test a trained BrainMGT model on a test dataset and evaluate its performance using accuracy, F1-score, and AUC.

- **Load the saved model.**
- **Preprocess the test fMRI data using the same pipeline as in training.**
- **Evaluate the model's performance on the test set.**

To use:
```bash
python test_brainmgt.py
```

## Dependencies

To run the BrainMGT project, you'll need the following Python libraries:

- **torch:** PyTorch for building neural networks
- **torch_geometric:** For Graph Attention Networks (GATs)
- **nibabel:** For loading and processing fMRI data
- **nipype:** For neuroimaging preprocessing
- **nilearn:** For working with neuroimaging data
- **pywavelets:** For Discrete Wavelet Transform (DWT)
- **scipy:** For statistical and signal processing
- **sklearn:** For evaluation metrics (accuracy, F1-score, AUC)
- **networkx:** For graph-based operations
- **community (python-louvain):** For community detection algorithms

You can install all dependencies via pip:

```bash
pip install torch torch-geometric nibabel nipype nilearn pywavelets scipy scikit-learn networkx python-louvain
```



## Citation:

````
Multi-scale Brain Graph Transformer for Brain Disease Diagnosis
````



## License

This project is licensed under the MIT License.
