# Hierarchical_Spatial_Scales.py

import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import networkx as nx
import community as community_louvain
import os

class HierarchicalSpatialScales:
    def __init__(self, preprocessed_fmri, output_dir):
        self.preprocessed_fmri = preprocessed_fmri
        self.output_dir = output_dir

        # Load the Schaefer 2018 7-network atlas with 1000 parcels (microscale)
        self.atlas = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
        self.labels = self.atlas['labels']

    def _extract_time_series(self):
        """
        Extract time series from the fMRI data using the Schaefer atlas.
        """
        masker = NiftiLabelsMasker(labels_img=self.atlas['maps'], standardize=True)
        time_series = masker.fit_transform(self.preprocessed_fmri)
        return time_series

    def microscale_network(self):
        """
        Define the microscale brain network with 1000 nodes.
        """
        time_series = self._extract_time_series()
        
        # Compute the correlation matrix as the connectivity measure
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        
        np.save(os.path.join(self.output_dir, 'microscale_correlation_matrix.npy'), correlation_matrix)
        return correlation_matrix

    def mesoscale_network(self, correlation_matrix, resolution=0.5):
        """
        Define the mesoscale brain network with approximately 500 communities using Louvain method.
        """
        # Create a graph from the correlation matrix
        graph = nx.Graph(correlation_matrix)
        
        # Apply Louvain method to detect communities
        partition = community_louvain.best_partition(graph, resolution=resolution)
        
        # Re-map nodes to their communities
        communities = np.zeros_like(list(partition.values()))
        for node, community in partition.items():
            communities[node] = community
        
        np.save(os.path.join(self.output_dir, 'mesoscale_communities.npy'), communities)
        return communities

    def macroscale_network(self, communities, num_macro_communities=100):
        """
        Define the macroscale brain network by aggregating mesoscale communities into about 100 larger communities.
        """
        unique_communities = np.unique(communities)
        num_communities = len(unique_communities)
        
        if num_communities <= num_macro_communities:
            raise ValueError("Mesoscale communities are fewer than the desired macroscale communities.")
        
        # Aggregate communities to form larger groups
        aggregation_factor = num_communities // num_macro_communities
        macro_communities = communities // aggregation_factor
        
        np.save(os.path.join(self.output_dir, 'macroscale_communities.npy'), macro_communities)
        return macro_communities

    def run_hierarchical_scales(self):
        """
        Execute the full pipeline to generate microscale, mesoscale, and macroscale networks.
        """
        # Microscale: 1000 nodes
        microscale_correlation_matrix = self.microscale_network()
        
        # Mesoscale: ~500 communities
        mesoscale_communities = self.mesoscale_network(microscale_correlation_matrix)
        
        # Macroscale: ~100 larger communities
        macroscale_communities = self.macroscale_network(mesoscale_communities)

# Example usage:
# hierarchical_scales = HierarchicalSpatialScales(preprocessed_fmri="path_to_preprocessed_fmri.nii.gz", output_dir="output_directory")
# hierarchical_scales.run_hierarchical_scales()
