import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import networkx as nx
import pickle


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for node classification with multiple layers and dropout.
    
    Args:
        input_dim (int): Dimension of input node features
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Number of output classes (3 for risk levels)
        num_layers (int): Number of GraphSAGE layers
        dropout (float): Dropout probability for regularization
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, num_layers=3, dropout=0.5):
        super(GraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through the GraphSAGE model.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
            edge_attr (torch.Tensor, optional): Edge attributes
            
        Returns:
            torch.Tensor: Node predictions
        """
        # Apply GraphSAGE layers with residual connections
        for i, conv in enumerate(self.convs[:-1]):
            x_new = conv(x, edge_index)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout_layer(x_new)
            x = x_new
            
        # Final layer without activation
        x = self.convs[-1](x, edge_index)
        
        return x


class VASPDataProcessor:
    """
    Data processor for VASP network data, handling conversion to PyTorch Geometric format.
    """
    
    def __init__(self, graph_path="vasp_graph.gpickle"):
        """
        Initialize the data processor.
        
        Args:
            graph_path (str): Path to the NetworkX graph pickle file
        """
        self.graph_path = graph_path
        self.graph = None
        self.data = None
        
    def load_graph(self):
        """Load the NetworkX graph from pickle file."""
        with open(self.graph_path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def process_features(self):
        """Process node and edge features for PyTorch Geometric."""
        if self.graph is None:
            raise ValueError("Graph not loaded. Call load_graph() first.")
            
        # Extract node features
        node_features = []
        node_labels = []
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            # Node features: [transaction_volume, degree, clustering_coefficient]
            degree = self.graph.degree(node)
            clustering = nx.clustering(self.graph, node)
            
            features = [
                node_data['transaction_volume'],
                degree,
                clustering
            ]
            
            node_features.append(features)
            node_labels.append(node_data['risk_level'])
            
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)
        
        # Normalize features
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
        
        # Extract edge information
        edge_index = []
        edge_attr = []
        
        for u, v, edge_data in self.graph.edges(data=True):
            edge_index.append([u, v])
            edge_index.append([v, u])  # Add reverse edge for undirected graph
            
            # Edge features: [amount, timestamp, fraud_ring_edge]
            edge_features = [
                edge_data['amount'],
                edge_data['timestamp'],
                edge_data['fraud_ring_edge']
            ]
            
            edge_attr.append(edge_features)
            edge_attr.append(edge_features)  # Same features for reverse edge
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Normalize edge features
        edge_attr = (edge_attr - edge_attr.mean(dim=0)) / (edge_attr.std(dim=0) + 1e-8)
        
        # Create PyTorch Geometric data object
        self.data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        print(f"Processed data: {self.data.x.shape[0]} nodes, {self.data.edge_index.shape[1]} edges")
        print(f"Node features shape: {self.data.x.shape}")
        print(f"Edge features shape: {self.data.edge_attr.shape}")
        
        return self.data


class VASPTrainer:
    """
    Trainer class for the VASP risk classification model.
    """
    
    def __init__(self, model, data, device='cpu'):
        """
        Initialize the trainer.
        
        Args:
            model (GraphSAGEModel): The GNN model
            data (Data): PyTorch Geometric data object
            device (str): Device to run training on
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        
        # Calculate class weights for imbalanced dataset
        unique, counts = np.unique(data.y.cpu().numpy(), return_counts=True)
        class_weights = torch.tensor(len(data.y) / (len(unique) * counts), dtype=torch.float).to(device)
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
    def create_data_splits(self, train_ratio=0.8, val_ratio=0.1, random_state=42):
        """
        Create train/validation/test splits.
        
        Args:
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            random_state (int): Random seed for reproducibility
        """
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        num_nodes = self.data.x.shape[0]
        indices = torch.randperm(num_nodes)
        
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        self.train_mask[indices[:train_size]] = True
        self.val_mask[indices[train_size:train_size + val_size]] = True
        self.test_mask[indices[train_size + val_size:]] = True
        
        print(f"Train: {self.train_mask.sum()} nodes")
        print(f"Validation: {self.val_mask.sum()} nodes")
        print(f"Test: {self.test_mask.sum()} nodes")
        
    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
        loss = self.criterion(out[self.train_mask], self.data.y[self.train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def evaluate(self, mask):
        """
        Evaluate the model on the given mask.
        
        Args:
            mask (torch.Tensor): Boolean mask for evaluation
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            pred = out[mask].max(1)[1]
            y_true = self.data.y[mask]
            
            # Calculate metrics
            accuracy = accuracy_score(y_true.cpu(), pred.cpu())
            f1 = f1_score(y_true.cpu(), pred.cpu(), average='macro')
            
            # ROC-AUC (one-vs-rest)
            probs = F.softmax(out[mask], dim=1)
            lb = LabelBinarizer()
            lb.fit(range(3))  # 3 classes
            y_true_bin = lb.transform(y_true.cpu())
            
            try:
                roc_auc = roc_auc_score(y_true_bin, probs.cpu(), average='macro', multi_class='ovr')
            except ValueError:
                roc_auc = 0.0  # Handle case where some classes are missing
            
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': pred.cpu().numpy(),
                'true_labels': y_true.cpu().numpy(),
                'probabilities': probs.cpu().numpy()
            }
            
    def train(self, epochs=200, early_stopping_patience=20):
        """
        Train the model with early stopping.
        
        Args:
            epochs (int): Maximum number of training epochs
            early_stopping_patience (int): Patience for early stopping
            
        Returns:
            dict: Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_roc_auc': []}
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_metrics = self.evaluate(self.val_mask)
            val_loss = self.criterion(
                self.model(self.data.x, self.data.edge_index, self.data.edge_attr)[self.val_mask],
                self.data.y[self.val_mask]
            ).item()
            
            # Update learning rate
            self.scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_metrics['f1_score'])
            history['val_roc_auc'].append(val_metrics['roc_auc'])
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'outputs/best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val F1: {val_metrics['f1_score']:.4f}, Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('outputs/best_model.pt'))
        
        return history


def compute_class_distribution(labels):
    """
    Compute and print class distribution.
    
    Args:
        labels (np.array): Array of class labels
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print("Class Distribution:")
    for class_id, count in zip(unique, counts):
        percentage = (count / total) * 100
        risk_level = ['Low', 'Medium', 'High'][class_id]
        print(f"  {risk_level} Risk (Class {class_id}): {count} ({percentage:.1f}%)")
