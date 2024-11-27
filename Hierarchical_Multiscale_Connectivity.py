# Hierarchical_Multiscale_Connectivity.py

import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform
from scipy.sparse import block_diag

class HierarchicalMultiscaleConnectivity:
    def __init__(self, preprocessed_fmri, hierarchical_scales, temporal_scales, output_dir):
        self.fmri_data = preprocessed_fmri
        self.hierarchical_scales = hierarchical_scales
        self.temporal_scales = temporal_scales
        self.output_dir = output_dir

    def intra_module_connectivity(self, scale, bold_signals):
        """
        Measure intra-module connectivity using Pearson correlation on BOLD signals.
        """
        num_nodes = bold_signals.shape[1]
        connectivity_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                connectivity_matrix[i, j] = pearsonr(bold_signals[:, i], bold_signals[:, j])[0]
                connectivity_matrix[j, i] = connectivity_matrix[i, j]

        np.save(f"{self.output_dir}/{scale}_intra_module_connectivity.npy", connectivity_matrix)
        return connectivity_matrix

    def construct_global_adjacency_matrix(self, scale, intra_module_matrices, num_nodes):
        """
        Construct global adjacency matrix by integrating module-specific connectivity matrices.
        """
        global_matrix = block_diag(intra_module_matrices).toarray()

        np.save(f"{self.output_dir}/{scale}_global_adjacency_matrix.npy", global_matrix)
        return global_matrix

    def inter_module_connectivity(self, scale, bold_signals, modules):
        """
        Measure inter-module connectivity by averaging BOLD signals across modules.
        """
        num_modules = len(modules)
        inter_module_matrix = np.zeros((num_modules, num_modules))

        for i in range(num_modules):
            module_i_signal = np.mean(bold_signals[:, modules[i]], axis=1)
            for j in range(i+1, num_modules):
                module_j_signal = np.mean(bold_signals[:, modules[j]], axis=1)
                inter_module_matrix[i, j] = pearsonr(module_i_signal, module_j_signal)[0]
                inter_module_matrix[j, i] = inter_module_matrix[i, j]

        np.save(f"{self.output_dir}/{scale}_inter_module_connectivity.npy", inter_module_matrix)
        return inter_module_matrix

    def differential_thresholding(self, global_matrix, intra_module_matrix, inter_module_matrix, intra_threshold=0.7, inter_threshold=0.3):
        """
        Apply differential thresholding to enhance intra-module connectivity and minimize inter-module connectivity.
        """
        # Apply thresholds
        enhanced_intra_matrix = np.where(intra_module_matrix > intra_threshold, intra_module_matrix, 0)
        reduced_inter_matrix = np.where(inter_module_matrix < inter_threshold, inter_module_matrix, 0)

        # Combine intra and inter matrices
        global_matrix[:enhanced_intra_matrix.shape[0], :enhanced_intra_matrix.shape[1]] = enhanced_intra_matrix
        global_matrix[enhanced_intra_matrix.shape[0]:, enhanced_intra_matrix.shape[1]:] = reduced_inter_matrix

        np.save(f"{self.output_dir}/differential_thresholded_matrix.npy", global_matrix)
        return global_matrix

    def generate_multiscale_network_dataset(self):
        """
        Process all fMRI datasets to generate a multiscale brain network dataset.
        """
        # Microscale: Intra-module connectivity estimation
        fast_bold_signals = self.temporal_scales['fast']
        microscale_intra_matrix = self.intra_module_connectivity('microscale', fast_bold_signals)

        # Mesoscale: Intra-module connectivity estimation
        intermediate_bold_signals = self.temporal_scales['intermediate']
        mesoscale_intra_matrix = self.intra_module_connectivity('mesoscale', intermediate_bold_signals)

        # Macroscale: Intra-module connectivity estimation
        slow_bold_signals = self.temporal_scales['slow']
        macroscale_intra_matrix = self.intra_module_connectivity('macroscale', slow_bold_signals)

        # Construct global adjacency matrices for each scale
        microscale_global_matrix = self.construct_global_adjacency_matrix('microscale', microscale_intra_matrix, num_nodes=1000)
        mesoscale_global_matrix = self.construct_global_adjacency_matrix('mesoscale', mesoscale_intra_matrix, num_nodes=500)
        macroscale_global_matrix = self.construct_global_adjacency_matrix('macroscale', macroscale_intra_matrix, num_nodes=100)

        # Inter-module connectivity estimation
        microscale_inter_matrix = self.inter_module_connectivity('microscale', fast_bold_signals, self.hierarchical_scales['microscale'])
        mesoscale_inter_matrix = self.inter_module_connectivity('mesoscale', intermediate_bold_signals, self.hierarchical_scales['mesoscale'])
        macroscale_inter_matrix = self.inter_module_connectivity('macroscale', slow_bold_signals, self.hierarchical_scales['macroscale'])

        # Apply differential thresholding for each scale
        self.differential_thresholding(microscale_global_matrix, microscale_intra_matrix, microscale_inter_matrix)
        self.differential_thresholding(mesoscale_global_matrix, mesoscale_intra_matrix, mesoscale_inter_matrix)
        self.differential_thresholding(macroscale_global_matrix, macroscale_intra_matrix, macroscale_inter_matrix)

# Example usage:
# hierarchical_scales = {
#     'microscale': np.load('path_to_microscale_data.npy'),
#     'mesoscale': np.load('path_to_mesoscale_data.npy'),
#     'macroscale': np.load('path_to_macroscale_data.npy')
# }
# temporal_scales = {
#     'fast': np.load('path_to_fast_frequency_data.npy'),
#     'intermediate': np.load('path_to_intermediate_frequency_data.npy'),
#     'slow': np.load('path_to_slow_frequency_data.npy')
# }
# connectivity = HierarchicalMultiscaleConnectivity(preprocessed_fmri="path_to_preprocessed_fmri_data.nii.gz", hierarchical_scales=hierarchical_scales, temporal_scales=temporal_scales, output_dir="output_directory")
# connectivity.generate_multiscale_network_dataset()
