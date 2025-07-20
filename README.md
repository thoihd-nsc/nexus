# Project Nexus GNN Assessment Submission

## Overview
This repository contains our solution to the Project Nexus GNN Technical Assessment. It demonstrates a scalable GraphSAGE model for VASP risk classification, with explainability via neighborhood visualization.

## Repository Structure
- `README.md`: This overview and setup guide
- `requirements.txt`: Required Python dependencies
- `generate_data.py`: Script to generate the synthetic VASP network dataset
- `src/`: Directory containing our source code
  - `model.py`: GraphSAGE model definition and data processing classes
  - `train.py`: Training and evaluation script with explainability functions
- `main.ipynb`: End-to-end Jupyter notebook demonstrating the complete solution
- `outputs/`: Contains generated results and model artifacts
  - `predictions.csv`: Test set predictions
  - `model.pt`: Trained model weights
  - `training_history.png`: Training progress visualization
  - `confusion_matrix.png`: Model performance matrix
  - `explanation_demo.png`: Model explanation visualization
- `setup.sh`: Environment setup script

## Setup and Execution

### Prerequisites
- Python 3.11
- CUDA-compatible GPU (optional, but recommended for faster training)

### Installation Steps

1. **Clone Repository**: 
   ```bash`
   git clone https://github.com/thoihd-nsc/nexus
   cd nexus
   ```

2. **Set Up Environment**: 
   ```bash
   chmod +x setup.sh
   bash setup.sh
   ```
   This will create a virtual environment and install all dependencies.

3. **Activate Environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Execute Notebook**: 
   Launch Jupyter and run all cells in `main.ipynb` to see the full workflow:
   ```bash
   jupyter notebook main.ipynb
   ```

5. **Review Results**: 
   - Key metrics (Accuracy, F1, ROC-AUC) are displayed in the notebook
   - All visualizations are saved to the `outputs/` directory
   - Model weights and predictions are saved for submission

## Key Features and Implementation

### Model Architecture
- **GraphSAGE**: 3-layer architecture with 128 hidden dimensions
- **Scalability**: Designed to handle large graphs through mini-batching
- **Node Features**: Transaction volume, degree centrality, clustering coefficient
- **Edge Features**: Transaction amount, timestamp, fraud ring indicator

### Class Imbalance Handling
- **Weighted Loss**: Class weights inversely proportional to frequency (70% Low, 20% Medium, 10% High)
- **Stratified Sampling**: Ensures balanced representation in train/val/test splits
- **Evaluation Metrics**: Focus on F1-Score (macro) and ROC-AUC for imbalanced performance

### Training Strategy
- **Early Stopping**: Patience of 20 epochs to prevent overfitting
- **Learning Rate Scheduling**: StepLR with gamma=0.5 every 50 epochs
- **Regularization**: Dropout (0.5) and weight decay (5e-4)
- **Batch Normalization**: Applied between GraphSAGE layers

### Model Explainability
- **Neighborhood Visualization**: Shows 2-hop neighbors with risk-based coloring
- **Prediction Analysis**: Detailed breakdown of model confidence for high-risk predictions
- **Structural Insights**: Highlights how graph topology influences risk assessment

## Expected Results
- **Target Performance**: ROC-AUC > 0.85 (as specified in requirements)
- **Training Time**: ~5-10 minutes on CPU, ~2-3 minutes on GPU

## Assumptions & Design Notes
- The solution was tested on Ubuntu 22.04 with Python 3.11
- GPU acceleration is optional but recommended for faster training
- Hyperparameters (learning rate: 0.01, hidden dim: 128, layers: 3) were selected based on empirical performance on the synthetic dataset
- The model uses full-batch training due to the manageable dataset size, but the architecture supports mini-batching for larger graphs
- Edge features (particularly `fraud_ring_edge`) are leveraged to enhance model performance and explainability

## Troubleshooting
- **Memory Issues**: Reduce batch size or hidden dimensions if encountering OOM errors
- **Slow Training**: Ensure PyTorch is using GPU if available (`torch.cuda.is_available()`)
- **Missing Dependencies**: Run `pip install -r requirements.txt` to install all packages
- **Visualization Issues**: Ensure matplotlib backend is properly configured for your system

## Technical Decisions
- **GraphSAGE over GCN**: Better scalability for large graphs through sampling
- **3-Layer Architecture**: Balances expressiveness with computational efficiency
- **Weighted Loss**: More effective than oversampling for this graph setting
- **Early Stopping**: Prevents overfitting while maintaining reproducibility

**Contact**: [Thoi Hoang - thoi.hd@nscsoftware.com] for any questions regarding this submission. 