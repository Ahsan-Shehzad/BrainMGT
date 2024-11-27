# train_brainmgt.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from fMRI_Preprocessing import fMRI_Preprocessing
from Hierarchical_Spatial_Scales import HierarchicalSpatialScales
from Multi_band_Temporal_Scaless import MultiBandTemporalScales
from Hierarchical_Multiscale_Connectivity import HierarchicalMultiscaleConnectivity
from Multi_scale_Brain_Graph_Transformers import MultiScaleBrainGraphTransformer, train_model

# Define paths
input_fmri_path = "path_to_fmri_data.nii.gz"
output_dir = "output_directory"

# Preprocess the fMRI data
preprocessing = fMRI_Preprocessing(input_file=input_fmri_path, output_dir=output_dir)
preprocessing.run_preprocessing()

# Load the preprocessed fMRI data
preprocessed_fmri_path = os.path.join(output_dir, "preprocessed_fmri.nii.gz")

# Generate hierarchical spatial scales
hierarchical_scales = HierarchicalSpatialScales(preprocessed_fmri=preprocessed_fmri_path, output_dir=output_dir)
hierarchical_scales.run_hierarchical_scales()

# Load the hierarchical spatial scales
microscale_path = os.path.join(output_dir, 'microscale_correlation_matrix.npy')
mesoscale_path = os.path.join(output_dir, 'mesoscale_communities.npy')
macroscale_path = os.path.join(output_dir, 'macroscale_communities.npy')

# Generate multi-band temporal scales
temporal_scales = MultiBandTemporalScales(preprocessed_fmri_data=preprocessed_fmri_path, hierarchical_scales={
    'microscale': np.load(microscale_path),
    'mesoscale': np.load(mesoscale_path),
    'macroscale': np.load(macroscale_path)
}, output_dir=output_dir)
temporal_scales.run_temporal_scales()

# Load the temporal scales
fast_frequency_path = os.path.join(output_dir, 'fast_frequency_band.npy')
intermediate_frequency_path = os.path.join(output_dir, 'intermediate_frequency_band.npy')
slow_frequency_path = os.path.join(output_dir, 'slow_frequency_band.npy')

# Generate hierarchical multiscale connectivity
connectivity = HierarchicalMultiscaleConnectivity(
    preprocessed_fmri=preprocessed_fmri_path,
    hierarchical_scales={
        'microscale': np.load(microscale_path),
        'mesoscale': np.load(mesoscale_path),
        'macroscale': np.load(macroscale_path)
    },
    temporal_scales={
        'fast': np.load(fast_frequency_path),
        'intermediate': np.load(intermediate_frequency_path),
        'slow': np.load(slow_frequency_path)
    },
    output_dir=output_dir
)
connectivity.generate_multiscale_network_dataset()

# Load the multi-scale brain network dataset (you can modify this part to load actual datasets)
# For simplicity, we are assuming that each dataset is already in Tensor format
microscale_data = torch.tensor(np.load(microscale_path), dtype=torch.float32)
mesoscale_data = torch.tensor(np.load(mesoscale_path), dtype=torch.float32)
macroscale_data = torch.tensor(np.load(macroscale_path), dtype=torch.float32)

# Placeholder for labels (to be replaced with actual labels)
labels = torch.randint(0, 2, (microscale_data.size(0),))  # Binary classification example

# Create a dataset combining the microscale, mesoscale, macroscale data and labels
dataset = TensorDataset(microscale_data, mesoscale_data, macroscale_data, labels)

# Split dataset into training, validation, and testing sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = MultiScaleBrainGraphTransformer(num_nodes=1000, num_heads=8, hidden_dim=128, output_dim=128, num_classes=2)

# Train the model
train_model(model, train_loader, val_loader, epochs=50, lr=0.001)

# After training, you can evaluate the model on the test set if desired
# evaluate_model(model, test_loader)
