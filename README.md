# Wasserstein-Rubinstein距离增强的图注意力专家融合模型

## 项目概述

本项目提出了一种基于Wasserstein-Rubinstein (WR) 距离的图注意力专家融合模型 (WR-EFM)，用于解决PubMed引文网络数据集上的节点分类问题。我们特别关注解决不同类别节点分类难度不平衡的问题，尤其是提高传统GCN模型在类别2上表现较弱的情况。

## 模型架构

我们的WR-EFM模型包含以下核心组件：

1. **专家模型层**
   - **GNN模型**：针对类别0和类别1优化的图卷积网络，增加了层归一化和残差连接
   - **Multi-hop GAT模型**：针对类别2优化的多跳图注意力网络，可以捕获更远距离的节点关系

2. **WR距离增强的融合机制**
   - 使用Wasserstein-Rubinstein距离度量不同模型表示之间的相似性
   - 动态调整融合权重，根据类别特性和模型置信度自适应分配权重
   - 全局类别平衡调整，确保预测分布与理想分布接近

3. **优化策略**
   - 类别特定增强因子，针对不同类别应用不同的增强系数
   - 高置信度模型优先策略，当某个模型对样本有高置信度时优先采用其预测
   - 共识增强机制，当不同模型达成一致预测时增强该预测的置信度

## 实验结果

在PubMed数据集上的实验结果表明，我们的WR-EFM模型在所有类别上都取得了平衡的性能：

| 模型 | 总体准确率 | 类别0 | 类别1 | 类别2 | 变异系数 |
|------|------------|-------|-------|-------|----------|
| GCN  | 79.8% | 80.0% | 84.0% | 74.4% | 0.058 |
| GNN (优化版) | 80.0% | 80.0% | 80.4% | 73.9% | 0.043 |
| Multi-hop GAT | 79.1% | 75.6% | 81.6% | 78.1% | 0.037 |
| 专家融合 | 79.0% | 77.8% | 78.0% | 78.4% | 0.004 |
| WR-EFM (本项目) | **80.1%** | 77.8% | 78.0% | **79.9%** | 0.013 |

WR-EFM模型的主要优势：
- 在类别2上实现了79.9%的准确率，比GCN提高了5.5%
- 总体准确率达到80.1%，超过了所有对比模型
- 变异系数显著低于GCN，表明各类别性能更加平衡

## 文件说明

- `pubmed-expert-fusion-wr.py`: WR距离增强的专家融合模型实现
- `pubmed-expert-fusion.py`: 基础专家融合模型实现
- `results/`: 存放实验结果和可视化图表

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric 2.0+
- NumPy
- Matplotlib
- Scikit-learn
- POT (Python Optimal Transport)

## 使用方法

1. 安装依赖：
```bash
pip install torch torch_geometric numpy matplotlib scikit-learn seaborn pot
```

2. 运行基础专家融合模型：
```bash
python pubmed-expert-fusion.py
```

3. 运行WR距离增强的专家融合模型：
```bash
python pubmed-expert-fusion-wr.py
```

## 模型参数说明

WR-EFM模型使用了以下关键参数：

- 类别权重分配：
  - 类别0: GNN 70%, GAT 30%
  - 类别1: GNN 88%, GAT 12%
  - 类别2: GNN 18%, GAT 82%

- 类别特定增强因子：
  - 类别0: 1.05
  - 类别1: 1.04
  - 类别2: 1.02

- 高置信度阈值: 0.85
- 共识类别增强系数: 1.10

## 可视化结果

运行模型后，会在`results/`目录下生成以下可视化结果：
- 混淆矩阵
- t-SNE节点嵌入可视化
- 各类别准确率条形图

## 贡献者

- 马梓航 - 3023209299

## 参考文献

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks.
2. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks.
3. Peyré, G., & Cuturi, M. (2019). Computational optimal transport. 