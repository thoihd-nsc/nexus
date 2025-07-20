import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def plot_training_history(history, save_path='outputs/training_history.png'):
    """
    Plot training history metrics.
    
    Args:
        history (dict): Training history dictionary
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation F1 Score
    axes[0, 1].plot(history['val_f1'], label='Validation F1', color='orange')
    axes[0, 1].set_title('Validation F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Validation ROC-AUC
    axes[1, 0].plot(history['val_roc_auc'], label='Validation ROC-AUC', color='green')
    axes[1, 0].set_title('Validation ROC-AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('ROC-AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # All metrics together
    axes[1, 1].plot(history['val_f1'], label='F1 Score', alpha=0.7)
    axes[1, 1].plot(history['val_roc_auc'], label='ROC-AUC', alpha=0.7)
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='outputs/confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Risk', 'Medium Risk', 'High Risk'],
                yticklabels=['Low Risk', 'Medium Risk', 'High Risk'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def explain_prediction(model, data, explainer, node_idx, save_path='outputs/explanation.png'):
    """
    Explain a single prediction using GNNExplainer.
    
    Args:
        model: Trained GNN model
        data: PyTorch Geometric data object
        explainer: GNNExplainer instance
        node_idx (int): Index of the node to explain
        save_path (str): Path to save the explanation plot
    """
    # Get explanation
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    
    # Get the subgraph for visualization
    node_mask = explanation.node_mask
    edge_mask = explanation.edge_mask
    
    # Handle node mask - it might be per feature, so we need to aggregate
    if node_mask.dim() > 1:
        # If node_mask is 2D (nodes x features), take the mean across features
        node_importance = node_mask.mean(dim=1)
    else:
        # If node_mask is 1D (nodes), use it directly
        node_importance = node_mask
    
    # Create a subgraph with important nodes and edges
    important_nodes = torch.where(node_importance > 0.1)[0].cpu().numpy()
    important_edges_idx = torch.where(edge_mask > 0.1)[0].cpu().numpy()
    
    # Create NetworkX graph for visualization
    G = nx.Graph()
    
    # Add nodes
    node_colors = []
    node_sizes = []
    
    for node in important_nodes:
        G.add_node(node)
        # Color based on risk level
        risk_level = data.y[node].item()
        if risk_level == 0:
            node_colors.append('lightblue')
        elif risk_level == 1:
            node_colors.append('orange')
        else:
            node_colors.append('red')
        
        # Size based on importance
        importance = node_importance[node].item()
        node_sizes.append(300 + importance * 1000)
    
    # Add edges
    edge_colors = []
    edge_widths = []
    
    for edge_idx in important_edges_idx:
        if edge_idx < len(data.edge_index[0]):
            u, v = data.edge_index[:, edge_idx].cpu().numpy()
            if u in important_nodes and v in important_nodes:
                G.add_edge(u, v)
                importance = edge_mask[edge_idx].item()
                edge_colors.append(importance)
                edge_widths.append(1 + importance * 5)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges
    if edge_colors and len(edge_colors) > 0:
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Add title and legend
    # Get model prediction safely
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        predicted_class = logits[node_idx].argmax().item()
    
    plt.title(f'GNN Explanation for Node {node_idx}\n'
              f'Predicted Risk: {["Low", "Medium", "High"][predicted_class]}\n'
              f'True Risk: {["Low", "Medium", "High"][data.y[node_idx].item()]}')
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='Low Risk'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=10, label='Medium Risk'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='High Risk')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Explanation visualization saved to {save_path}")


def save_predictions(test_metrics, test_mask, save_path='outputs/predictions.csv'):
    """
    Save test predictions to CSV file.
    
    Args:
        test_metrics (dict): Test evaluation metrics
        test_mask (torch.Tensor): Test mask
        save_path (str): Path to save predictions
    """
    test_indices = torch.where(test_mask)[0].cpu().numpy()
    
    df = pd.DataFrame({
        'node_id': test_indices,
        'true_label': test_metrics['true_labels'],
        'predicted_label': test_metrics['predictions'],
        'prob_low_risk': test_metrics['probabilities'][:, 0],
        'prob_medium_risk': test_metrics['probabilities'][:, 1],
        'prob_high_risk': test_metrics['probabilities'][:, 2]
    })
    
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def create_simple_visualization(data, node_idx, save_path='outputs/simple_explanation.png'):
    """
    Create a simple visualization of the node's neighborhood.
    
    Args:
        data: PyTorch Geometric data object
        node_idx (int): Index of the node to visualize
        save_path (str): Path to save the visualization
    """
    # Get 2-hop neighborhood
    edge_index = data.edge_index.cpu().numpy()
    
    # Find neighbors
    neighbors = set()
    neighbors.add(node_idx)
    
    # 1-hop neighbors
    for i, (u, v) in enumerate(edge_index.T):
        if u == node_idx:
            neighbors.add(v)
        elif v == node_idx:
            neighbors.add(u)
    
    # 2-hop neighbors (limited to prevent overcrowding)
    first_hop = list(neighbors)
    for node in first_hop[:5]:  # Limit to first 5 neighbors
        for i, (u, v) in enumerate(edge_index.T):
            if u == node and len(neighbors) < 20:
                neighbors.add(v)
            elif v == node and len(neighbors) < 20:
                neighbors.add(u)
    
    # Create subgraph
    G = nx.Graph()
    node_colors = []
    node_sizes = []
    
    for node in neighbors:
        G.add_node(node)
        risk_level = data.y[node].item()
        if risk_level == 0:
            node_colors.append('lightblue')
        elif risk_level == 1:
            node_colors.append('orange')
        else:
            node_colors.append('red')
        
        # Highlight target node
        if node == node_idx:
            node_sizes.append(800)
        else:
            node_sizes.append(300)
    
    # Add edges
    for i, (u, v) in enumerate(edge_index.T):
        if u in neighbors and v in neighbors:
            G.add_edge(u, v)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f'Neighborhood of Node {node_idx}\n'
              f'Risk Level: {["Low", "Medium", "High"][data.y[node_idx]]}')
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='Low Risk'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=10, label='Medium Risk'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='High Risk')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Simple visualization saved to {save_path}")
