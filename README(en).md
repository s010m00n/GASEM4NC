# Wasserstein-Rubinstein Distance Enhanced Graph Attention Expert Fusion Model

## Project Overview

This project introduces a Wasserstein-Rubinstein (WR) distance enhanced Graph Attention Expert Fusion Model (WR-EFM) for node classification on the PubMed citation network dataset. We specifically address the performance imbalance across different node categories, with particular focus on improving the traditionally weak performance of GCN models on Category 2 nodes.

## Model Architecture

Our WR-EFM model consists of the following core components:

1. **Expert Model Layer**
   - **GNN Model**: A graph convolutional network optimized for Categories 0 and 1, enhanced with layer normalization and residual connections
   - **Multi-hop GAT Model**: A multi-hop graph attention network optimized for Category 2, capable of capturing longer-range node relationships

2. **WR Distance Enhanced Fusion Mechanism**
   - Uses Wasserstein-Rubinstein distance to measure similarity between representations from different models
   - Dynamically adjusts fusion weights based on category-specific characteristics and model confidence
   - Applies global class balance adjustment to ensure prediction distribution aligns with ideal distribution

3. **Optimization Strategies**
   - Category-specific enhancement factors applied to different classes
   - High-confidence model prioritization when a model shows strong confidence in its prediction
   - Consensus enhancement mechanism to boost predictions when models agree

## Experimental Results

Experiments on the PubMed dataset demonstrate that our WR-EFM model achieves balanced performance across all categories:

| Model | Overall Accuracy | Category 0 | Category 1 | Category 2 | CV |
|-------|-----------------|-----------|-----------|-----------|-----|
| GCN   | 79.8% | 80.0% | 84.0% | 74.4% | 0.058 |
| GNN (Optimized) | 80.0% | 80.0% | 80.4% | 73.9% | 0.043 |
| Multi-hop GAT | 79.1% | 75.6% | 81.6% | 78.1% | 0.037 |
| Expert Fusion | 79.0% | 77.8% | 78.0% | 78.4% | 0.004 |
| WR-EFM (Ours) | **80.1%** | 77.8% | 78.0% | **79.9%** | 0.013 |

Key advantages of the WR-EFM model:
- Achieves 79.9% accuracy on Category 2, a 5.5% improvement over GCN
- Reaches 80.1% overall accuracy, outperforming all comparison models
- Significantly lower coefficient of variation than GCN, indicating more balanced performance across categories

## File Description

- `pubmed-expert-fusion-wr.py`: Implementation of the WR distance enhanced expert fusion model
- `pubmed-expert-fusion.py`: Implementation of the basic expert fusion model
- `results/`: Directory containing experimental results and visualizations

## Environment Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric 2.0+
- NumPy
- Matplotlib
- Scikit-learn
- POT (Python Optimal Transport)

## Usage Instructions

1. Install dependencies:
```bash
pip install torch torch_geometric numpy matplotlib scikit-learn seaborn pot
```

2. Run the basic expert fusion model:
```bash
python pubmed-expert-fusion.py
```

3. Run the WR distance enhanced expert fusion model:
```bash
python pubmed-expert-fusion-wr.py
```

## Model Parameter Details

The WR-EFM model uses the following key parameters:

- Category weight distribution:
  - Category 0: GNN 70%, GAT 30%
  - Category 1: GNN 88%, GAT 12%
  - Category 2: GNN 18%, GAT 82%

- Category-specific enhancement factors:
  - Category 0: 1.05
  - Category 1: 1.04
  - Category 2: 1.02

- High confidence threshold: 0.85
- Consensus category enhancement coefficient: 1.10

## Visualization Results

After running the model, the following visualization results will be generated in the `results/` directory:
- Confusion matrix
- t-SNE node embedding visualization
- Category accuracy bar chart

## Contributors

- Zihang MA - 3023209299

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks.
2. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks.
3. Peyré, G., & Cuturi, M. (2019). Computational optimal transport. 