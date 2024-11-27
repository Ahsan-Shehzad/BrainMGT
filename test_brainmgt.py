# test_brainmgt.py

import os
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from fMRI_Preprocessing import fMRI_Preprocessing
from Hierarchical_Spatial_Scales import HierarchicalSpatialScales
from Multi_band_Temporal_Scaless import MultiBandTemporalScales
from Hierarchical_Multiscale_Connectivity import HierarchicalMultiscaleConnectivity
from Multi_scale_Brain_Graph_Transformers import MultiScaleBrainGraphTransformer

# Define paths
input_fmri_path = "path_to_test_fmri_data.nii.gz"
model_path = "path_to_saved_model.pth"
output_dir = "output_directory"

# Preprocess the test fMRI data
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

# Load the multi-scale brain network dataset (replace this part with actual test dataset loading)
microscale_data = torch.tensor(np.load(microscale_path), dtype=torch.float32)
mesoscale_data = torch.tensor(np.load(mesoscale_path), dtype=torch.float32)
macroscale_data = torch.tensor(np.load(macroscale_path), dtype=torch.float32)

# Placeholder for test labels (replace this with actual labels)
test_labels = torch.randint(0, 2, (microscale_data.size(0),))

# Create a test dataset and DataLoader
test_dataset = TensorDataset(microscale_data, mesoscale_data, macroscale_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model
model = MultiScaleBrainGraphTransformer(num_nodes=1000, num_heads=8, hidden_dim=128, output_dim=128, num_classes=2)
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluate the model on the test set
all_preds = []
all_labels = []

with torch.no_grad():
    for microscale, mesoscale, macroscale, labels in test_loader:
        outputs = model(microscale, mesoscale, macroscale)
        _, preds = torch.max(outputs, 1)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Concatenate all predictions and labels
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Calculate evaluation metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_preds)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1-score: {f1:.4f}")
print(f"Test AUC: {auc:.4f}")
