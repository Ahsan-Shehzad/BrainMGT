# Multi_scale_Brain_Graph_Transformers.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch.nn.functional import cross_entropy, softmax

class MultiScaleBrainGraphTransformer(nn.Module):
    def __init__(self, num_nodes, num_heads, hidden_dim, output_dim, num_classes):
        super(MultiScaleBrainGraphTransformer, self).__init__()
        
        # GAT Layers for each scale (Microscale, Mesoscale, Macroscale)
        self.gat_micro = GATConv(in_channels=num_nodes, out_channels=hidden_dim, heads=num_heads)
        self.gat_meso = GATConv(in_channels=num_nodes, out_channels=hidden_dim, heads=num_heads)
        self.gat_macro = GATConv(in_channels=num_nodes, out_channels=hidden_dim, heads=num_heads)

        # Learnable Positional Encoding for nodes
        self.positional_encoding = nn.Embedding(num_nodes, hidden_dim)
        
        # Self-attention for scale-specific encoders
        self.scale_specific_attention_micro = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.scale_specific_attention_meso = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.scale_specific_attention_macro = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        # Cross-scale Encoders
        self.cross_scale_encoder_micro_meso = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.cross_scale_encoder_meso_macro = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        # Hierarchical Adaptive Fusion
        self.adaptive_fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, microscale, mesoscale, macroscale):
        # Generate embeddings with GAT layers
        micro_embeddings = self.gat_micro(microscale.x, microscale.edge_index)
        meso_embeddings = self.gat_meso(mesoscale.x, mesoscale.edge_index)
        macro_embeddings = self.gat_macro(macroscale.x, macroscale.edge_index)
        
        # Positional encoding
        positional_micro = self.positional_encoding(microscale.pos)
        positional_meso = self.positional_encoding(mesoscale.pos)
        positional_macro = self.positional_encoding(macroscale.pos)
        
        # Add positional encodings to node embeddings
        micro_embeddings += positional_micro
        meso_embeddings += positional_meso
        macro_embeddings += positional_macro
        
        # Scale-specific encoders (Self-attention)
        micro_encoded, _ = self.scale_specific_attention_micro(micro_embeddings, micro_embeddings, micro_embeddings)
        meso_encoded, _ = self.scale_specific_attention_meso(meso_embeddings, meso_embeddings, meso_embeddings)
        macro_encoded, _ = self.scale_specific_attention_macro(macro_embeddings, macro_embeddings, macro_embeddings)
        
        # Cross-scale encoders (Cross-attention)
        micro_meso_cross_encoded, _ = self.cross_scale_encoder_micro_meso(micro_encoded, meso_encoded, meso_encoded)
        meso_macro_cross_encoded, _ = self.cross_scale_encoder_meso_macro(meso_encoded, macro_encoded, macro_encoded)
        
        # Hierarchical Adaptive Fusion
        fused_features = torch.cat((micro_meso_cross_encoded, meso_macro_cross_encoded, macro_encoded), dim=1)
        fused_features = self.adaptive_fusion(fused_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for microscale, mesoscale, macroscale, labels in train_loader:
            optimizer.zero_grad()
            output = model(microscale, mesoscale, macroscale)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for microscale, mesoscale, macroscale, labels in val_loader:
            output = model(microscale, mesoscale, macroscale)
            loss = criterion(output, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    accuracy = correct_predictions / total_predictions
    return total_loss, accuracy

# Example usage:
# Define your dataset loaders (train_loader, val_loader) with preprocessed multi-scale brain networks
# model = MultiScaleBrainGraphTransformer(num_nodes=1000, num_heads=8, hidden_dim=128, output_dim=128, num_classes=2)
# train_model(model, train_loader, val_loader, epochs=50, lr=0.001)
