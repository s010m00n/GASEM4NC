import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops, remove_self_loops, to_dense_adj, dense_to_sparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
plt.rcParams["font.size"]=16
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载PubMed数据集
dataset = Planetoid(root="data/PubMed", name="PubMed", transform=NormalizeFeatures())
data = dataset[0].to(device)

print(f"数据集：{data}")
print(f"节点特征维度: {data.x.shape}")
print(f"节点数量: {data.num_nodes}")
print(f"边数量: {data.num_edges}")
print(f"特征维度: {dataset.num_features}")
print(f"类别数量: {dataset.num_classes}")

# 统计每个类别的样本数量
class_counts = torch.bincount(data.y)
print(f"类别分布: {class_counts}")

# 创建间接连接的边
def create_higher_order_adj(edge_index, num_nodes, max_hop=6):
    """
    创建高阶邻接矩阵，考虑最多max_hop步的连接
    """
    # 创建原始的邻接矩阵
    adj = to_dense_adj(edge_index)[0]
    
    # 初始化高阶邻接矩阵
    higher_order_adjs = [adj]  # 1-hop (原始边)
    
    # 计算2到max_hop步的邻接矩阵
    last_adj = adj
    for i in range(1, max_hop):
        next_adj = torch.matmul(last_adj, adj)
        # 确保对角线为0（去除自环）
        next_adj.fill_diagonal_(0)
        # 二值化
        next_adj = (next_adj > 0).float()
        higher_order_adjs.append(next_adj)
        last_adj = next_adj
        
    return higher_order_adjs

# 可视化函数
def visualize_comparison(h, true_labels, pred_labels):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    
    plt.figure(figsize=(20, 10))
    
    # 左侧子图：真实标签
    plt.subplot(1, 2, 1)
    plt.title('真实标签')
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=true_labels.cpu().numpy(), cmap="Set2")
    
    # 右侧子图：预测标签
    plt.subplot(1, 2, 2)
    plt.title('预测标签')
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=pred_labels.cpu().numpy(), cmap="Set2")
    
    plt.savefig('results/pubmed_expert_fusion_comparison.png')
    plt.show()

# 绘制准确率条形图
def plot_accuracy_bar(class_accs, overall_acc, title='各类别准确率'):
    class_labels = [f'类别{i}' for i in range(len(class_accs))]
    class_labels.append('总体')
    
    all_accs = class_accs + [overall_acc]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_labels, all_accs, color=['#3274A1', '#E1812C', '#3A923A'])
    
    # 在柱状图顶部添加准确率数值
    for bar, acc in zip(bars, all_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12)
    
    plt.ylim(0, 1.0)
    plt.ylabel('准确率')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('results/pubmed_expert_fusion_accuracy.png')
    plt.show()

# 定义带有层归一化和残差连接的GNN模型
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)  # 降低dropout率，防止过度正则化
        
        # 添加层归一化提高训练稳定性
        self.layer_norm1 = torch.nn.LayerNorm(hidden_channels)
        
        # 添加用于残差连接的线性投影
        self.skip_proj = torch.nn.Linear(dataset.num_features, hidden_channels)

    def forward(self, x, edge_index, higher_order_adjs=None):
        # 保存输入用于残差连接
        input_x = x
        
        # 第一层卷积
        x = self.conv1(x, edge_index)
        
        # 先应用层归一化
        x = self.layer_norm1(x)
        
        # 带权重的残差连接 - 给原始特征更少的权重
        x_residual = self.skip_proj(input_x)
        x = 0.8 * x + 0.2 * x_residual  # 80%卷积特征，20%原始特征
        
        # 激活函数和dropout
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第二层卷积
        x = self.conv2(x, edge_index)
        
        # 不添加log_softmax以保持与其他模型一致
        return x, None  # 返回None作为第二个元素，保持接口一致

# 多跳GAT模型
class MultiHopGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, max_hop=2, dropout=0.5, heads=2):
        super(MultiHopGAT, self).__init__()
        
        self.max_hop = max_hop
        self.dropout = dropout
        self.heads = heads
        
        # 计算每个层的特征维度
        self.hidden_per_hop = hidden_channels // max_hop
        
        # 对每个跳数使用一个GAT卷积层
        self.conv1_list = nn.ModuleList()
        for _ in range(max_hop):
            self.conv1_list.append(GATConv(in_channels, self.hidden_per_hop, heads=heads))
        
        # 第二层GAT卷积 - 注意这里的输入维度是 hidden_per_hop * heads
        self.conv2 = GATConv(self.hidden_per_hop * heads, out_channels, heads=1, concat=False)
        
        # 用于不同跳数间的注意力权重 - 使用更好的初始化
        initial_weights = torch.tensor([0.7, 0.3][:max_hop])
        self.hop_attention = nn.Parameter(initial_weights)
        
        # 添加层归一化(LayerNorm)提高训练稳定性
        self.layer_norm1 = nn.LayerNorm(self.hidden_per_hop * heads)
        self.layer_norm2 = nn.LayerNorm(out_channels)
        
        # 添加用于残差连接的线性投影
        self.skip_proj = nn.Linear(in_channels, self.hidden_per_hop * heads)
        self.final_proj = nn.Linear(self.hidden_per_hop * heads, out_channels)
        
    def forward(self, x, edge_index, higher_order_adjs=None):
        # 保存输入用于残差连接
        input_x = x
        
        if higher_order_adjs is None:
            # 如果没有提供高阶邻接矩阵，只使用原始边索引
            x_hop = self.conv1_list[0](x, edge_index)
            x_hop = F.dropout(x_hop, p=self.dropout, training=self.training)
            x_hop = F.elu(x_hop)
            x_combined = x_hop
        else:
            # 对每个跳数的边应用GAT卷积
            x_per_hop = []
            for i in range(self.max_hop):
                # 对于每个跳数，获取对应的边索引
                if i == 0:
                    curr_edge_index = edge_index
                else:
                    # 将高阶邻接矩阵转换为边索引
                    curr_edge_index = dense_to_sparse(higher_order_adjs[i-1])[0]
                    
                # 应用GAT卷积 (会自动concat不同头的结果)
                x_hop = self.conv1_list[i](x, curr_edge_index)
                # 每个hop后应用dropout
                x_hop = F.dropout(x_hop, p=self.dropout, training=self.training)
                x_hop = F.elu(x_hop)
                x_per_hop.append(x_hop)
            
            # 计算不同跳数间的注意力权重
            hop_attn_weights = F.softmax(self.hop_attention, dim=0)
            
            # 加权合并不同跳数的特征
            x_combined = torch.zeros_like(x_per_hop[0])
            for i in range(self.max_hop):
                x_combined += hop_attn_weights[i] * x_per_hop[i]
        
        # 第一层残差连接
        x_residual = self.skip_proj(input_x)
        x_combined = x_combined + x_residual
        
        # 应用层归一化
        x_combined = self.layer_norm1(x_combined)
        
        # 应用dropout
        x_combined = F.dropout(x_combined, p=self.dropout, training=self.training)
        
        # 保存第一层结果用于第二层残差连接
        first_layer_output = x_combined
        
        # 第二层卷积
        x_combined = self.conv2(x_combined, edge_index)
        
        # 第二层残差连接
        x_residual2 = self.final_proj(first_layer_output)
        x_combined = x_combined + x_residual2
        
        # 应用第二层归一化
        x_combined = self.layer_norm2(x_combined)
        
        # 返回结果和注意力权重(如果有高阶邻接矩阵)
        if higher_order_adjs is None:
            return x_combined
        else:
            return x_combined, hop_attn_weights

# 创建高阶邻接矩阵 (考虑1-2跳)
MAX_HOP = 2
higher_order_adjs = create_higher_order_adj(data.edge_index, data.num_nodes, max_hop=MAX_HOP)
print(f"创建了 {len(higher_order_adjs)} 个高阶邻接矩阵")

# 统计每个跳数的边的数量
for i, adj in enumerate(higher_order_adjs):
    if i == 0:
        print(f"{i+1}跳连接数: {data.edge_index.shape[1]}")
    else:
        edge_index = dense_to_sparse(adj)[0]
        print(f"{i+1}跳连接数: {edge_index.shape[1]}")

# 创建模型实例 - 确保与预训练模型结构一致
hidden_channels = 64  # MultiHopGAT模型使用的隐藏层维度
gnn_hidden_channels = 512  # GNN模型使用的隐藏层维度

# 创建GNN模型
gnn_model = GNN(
    hidden_channels=gnn_hidden_channels
).to(device)

# 创建attn2模型(用于预测class2)
attn2_model = MultiHopGAT(
    in_channels=dataset.num_features, 
    hidden_channels=hidden_channels,
    out_channels=dataset.num_classes,
    max_hop=MAX_HOP,
    dropout=0.3,
    heads=2
).to(device)

# 加载预训练模型权重
print("加载预训练模型权重...")

# 加载GNN模型权重
try:
    gnn_model.load_state_dict(torch.load('results/best_gnn_model.pt'))
    print("已加载GNN模型: results/best_gnn_model.pt")
    use_attn_for_gnn = False
except Exception as e:
    print(f"加载GNN模型失败: {e}")
    use_attn_for_gnn = True

# 加载attn2模型权重
try:
    attn2_model.load_state_dict(torch.load('results/best_attn_model_class2.pt'))
    print("已加载attn2模型: results/best_attn_model_class2.pt")
except Exception as e:
    print(f"加载attn2模型失败: {e}")

# 所有模型设为评估模式
gnn_model.eval()
attn2_model.eval()

# 融合预测函数
def expert_fusion_predict():
    with torch.no_grad():
        # GNN模型预测
        gnn_logits, _ = gnn_model(data.x, data.edge_index, higher_order_adjs)
        
        # attn2模型预测
        attn2_logits, _ = attn2_model(data.x, data.edge_index, higher_order_adjs)
        
        # 应用softmax得到概率
        gnn_probs = F.softmax(gnn_logits, dim=1)
        attn2_probs = F.softmax(attn2_logits, dim=1)
        
        # 创建融合结果的张量
        combined_probs = torch.zeros_like(gnn_probs)
        
        # 融合策略：
        # - 类别0和类别1由GNN模型负责
        # - 类别2由attn2模型负责
        
        # 类别0由GNN负责 (权重0.95)，辅以attn2 (权重0.05)
        combined_probs[:, 0] = 0.95 * gnn_probs[:, 0] + 0.05 * attn2_probs[:, 0]
        
        # 类别1由GNN负责 (权重0.95)，辅以attn2 (权重0.05)
        combined_probs[:, 1] = 0.95 * gnn_probs[:, 1] + 0.05 * attn2_probs[:, 1]
        
        # 类别2由attn2负责 (权重0.8)，辅以GNN (权重0.2)
        combined_probs[:, 2] = 0.2 * gnn_probs[:, 2] + 0.8 * attn2_probs[:, 2]
        
        # 重新归一化确保每一行概率之和为1
        row_sums = combined_probs.sum(dim=1, keepdim=True)
        combined_probs = combined_probs / row_sums
        
        # 获取最终预测
        pred = combined_probs.argmax(dim=1)
        
        return pred, combined_probs

# 自适应融合预测函数
def adaptive_fusion_predict():
    with torch.no_grad():
        # GNN模型预测
        gnn_logits, _ = gnn_model(data.x, data.edge_index, higher_order_adjs)
        
        # attn2模型预测
        attn2_logits, _ = attn2_model(data.x, data.edge_index, higher_order_adjs)
        
        # 应用softmax得到概率
        gnn_probs = F.softmax(gnn_logits, dim=1)
        attn2_probs = F.softmax(attn2_logits, dim=1)
        
        # 获取每个模型对每个样本预测的最高置信度
        gnn_conf, _ = torch.max(gnn_probs, dim=1)
        attn2_conf, _ = torch.max(attn2_probs, dim=1)
        
        # 堆叠所有置信度，准备计算权重
        all_conf = torch.stack([gnn_conf, attn2_conf], dim=1)
        
        # 使用softmax计算每个模型的权重
        model_weights = F.softmax(all_conf * 2.0, dim=1)  # 乘以2.0使得差异更加明显
        
        # 创建融合结果的张量
        combined_probs = torch.zeros_like(gnn_probs)
        
        # 对每个样本，根据置信度动态融合各个模型的预测
        for i in range(data.num_nodes):
            # 每个模型对当前样本的权重
            w_gnn, w_attn2 = model_weights[i]
            
            # 对每个类别，计算加权概率
            for c in range(dataset.num_classes):
                # 基础权重
                base_weights = {
                    0: [0.95, 0.05],  # 类别0: [gnn, attn2]
                    1: [0.95, 0.05],  # 类别1: [gnn, attn2]
                    2: [0.2, 0.8],  # 类别2: [gnn, attn2]
                }.get(c, [0.5, 0.5])  # 默认权重
                
                # 动态调整的权重 - 结合基础权重和置信度
                b_gnn, b_attn2 = base_weights
                adj_w_gnn = b_gnn * (1.0 + w_gnn)
                adj_w_attn2 = b_attn2 * (1.0 + w_attn2)
                
                # 归一化
                sum_w = adj_w_gnn + adj_w_attn2
                adj_w_gnn /= sum_w
                adj_w_attn2 /= sum_w
                
                # 计算该样本该类别的融合概率
                combined_probs[i, c] = adj_w_gnn * gnn_probs[i, c] + adj_w_attn2 * attn2_probs[i, c]
        
        # 获取最终预测
        pred = combined_probs.argmax(dim=1)
        
        return pred, combined_probs

# 比较不同融合策略
def compare_fusion_strategies():
    # 原始融合策略
    fixed_pred, fixed_probs = expert_fusion_predict()
    
    # 自适应融合策略
    adaptive_pred, adaptive_probs = adaptive_fusion_predict()
    
    # 计算验证集准确率
    fixed_correct = (fixed_pred[data.val_mask] == data.y[data.val_mask]).sum().item()
    adaptive_correct = (adaptive_pred[data.val_mask] == data.y[data.val_mask]).sum().item()
    val_total = int(data.val_mask.sum())
    
    fixed_acc = fixed_correct / val_total
    adaptive_acc = adaptive_correct / val_total
    
    # 计算每个类别在验证集上的准确率
    fixed_class_accs = []
    adaptive_class_accs = []
    for c in range(dataset.num_classes):
        mask = (data.y == c) & data.val_mask
        if mask.sum() > 0:
            # 固定策略
            fixed_correct_c = (fixed_pred[mask] == data.y[mask]).sum().item()
            fixed_acc_c = fixed_correct_c / int(mask.sum())
            fixed_class_accs.append(fixed_acc_c)
            
            # 自适应策略
            adaptive_correct_c = (adaptive_pred[mask] == data.y[mask]).sum().item()
            adaptive_acc_c = adaptive_correct_c / int(mask.sum())
            adaptive_class_accs.append(adaptive_acc_c)
        else:
            fixed_class_accs.append(0.0)
            adaptive_class_accs.append(0.0)
    
    print(f"\n验证集融合策略比较:")
    print(f"固定权重融合准确率: {fixed_acc:.4f}")
    print(f"自适应融合准确率: {adaptive_acc:.4f}")
    
    print("\n各类别验证集准确率比较:")
    print("┌─────────────┬───────────┬───────────┬───────────┐")
    print("│     策略    │   类别0   │   类别1   │   类别2   │")
    print("├─────────────┼───────────┼───────────┼───────────┤")
    print(f"│  固定权重   │   {fixed_class_accs[0]:.4f}  │   {fixed_class_accs[1]:.4f}  │   {fixed_class_accs[2]:.4f}  │")
    print(f"│  自适应权重 │   {adaptive_class_accs[0]:.4f}  │   {adaptive_class_accs[1]:.4f}  │   {adaptive_class_accs[2]:.4f}  │")
    print("└─────────────┴───────────┴───────────┴───────────┘")
    
    # 返回更好的策略的预测结果
    if adaptive_acc > fixed_acc:
        print("选择自适应融合策略")
        return adaptive_pred, adaptive_probs
    else:
        print("选择固定权重融合策略")
        return fixed_pred, fixed_probs

# 评估函数
def evaluate(mask, pred=None, probs=None):
    if pred is None or probs is None:
        # 如果没有提供预测结果，则使用最佳融合策略
        pred, probs = compare_fusion_strategies()
        
    correct = (pred[mask] == data.y[mask]).sum().item()
    total = int(mask.sum())
    accuracy = correct / total
    
    # 计算每个类别的准确率和样本数量
    class_accs = []
    class_counts = []
    for c in range(dataset.num_classes):
        class_mask = (data.y == c) & mask
        class_total = int(class_mask.sum().item())
        class_counts.append(class_total)
        
        if class_total > 0:
            class_correct = (pred[class_mask] == data.y[class_mask]).sum().item()
            class_accuracy = class_correct / class_total
            class_accs.append(class_accuracy)
        else:
            class_accs.append(0.0)
            
    return accuracy, class_accs, pred[mask], data.y[mask], class_counts

# 评估模型性能
print("\n开始评估专家融合模型性能...")

# 选择更好的融合策略
global_pred, global_probs = compare_fusion_strategies()

# 计算训练、验证和测试集准确率
train_acc, train_class_accs, train_pred, train_true, train_counts = evaluate(data.train_mask, global_pred, global_probs)
val_acc, val_class_accs, val_pred, val_true, val_counts = evaluate(data.val_mask, global_pred, global_probs)
test_acc, test_class_accs, test_pred, test_true, test_counts = evaluate(data.test_mask, global_pred, global_probs)

# 打印结果
print(f"训练集准确率: {train_acc:.4f}")
print(f"验证集准确率: {val_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

# 打印每个类别的准确率和样本数量 - 格式化输出
print("\n各类别准确率:")
print("┌───────────┬───────────┬────────────┐")
print("│   类别    │   准确率  │  样本数量  │")
print("├───────────┼───────────┼────────────┤")
for i, acc in enumerate(test_class_accs):
    print(f"│   类别 {i}  │   {acc:.4f}  │     {test_counts[i]:<6} │")
print("└───────────┴───────────┴────────────┘")

# 显示模型比较
print("\n模型在各类别上的表现比较:")
print("┌───────────────┬───────────┬───────────┬───────────┐")
print("│     模型      │   类别0   │   类别1   │   类别2   │")
print("├───────────────┼───────────┼───────────┼───────────┤")
print(f"│ 专家融合(当前) │   {test_class_accs[0]:.4f}  │   {test_class_accs[1]:.4f}  │   {test_class_accs[2]:.4f}  │")
print("└───────────────┴───────────┴───────────┴───────────┘")

# 绘制准确率条形图
print("\n生成准确率条形图...")
plot_accuracy_bar(test_class_accs, test_acc, title='PubMed各类别测试准确率')

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, title='混淆矩阵'):
    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('results/pubmed_expert_fusion_confusion_matrix.png')
    plt.show()

# 计算并绘制测试集混淆矩阵
print("\n生成混淆矩阵...")
plot_confusion_matrix(data.y[data.test_mask], global_pred[data.test_mask], title='测试集混淆矩阵')

# 可视化节点嵌入和预测结果
def visualize_embeddings():
    # 使用不同模型的嵌入结果
    with torch.no_grad():
        # 获取各模型的嵌入
        gnn_embed, _ = gnn_model(data.x, data.edge_index, higher_order_adjs)
        attn2_embed, _ = attn2_model(data.x, data.edge_index, higher_order_adjs)
            
        # 将嵌入拼接起来
        combined_embed = torch.cat([
            gnn_embed, 
            attn2_embed
        ], dim=1)
        
        # 分析各类别节点的预测情况
        pred = global_pred
        for c in range(dataset.num_classes):
            # 找到类别c的节点在测试集中的索引
            class_mask = (data.y == c) & data.test_mask
            total = int(class_mask.sum().item())
            correct = (pred[class_mask] == data.y[class_mask]).sum().item()
            accuracy = correct / total if total > 0 else 0
            
            # 分析错误预测的去向
            if total > 0 and correct < total:
                misclassified = class_mask & (pred != data.y)
                pred_counts = {}
                for i in range(dataset.num_classes):
                    count = (pred[misclassified] == i).sum().item()
                    if count > 0:
                        pred_counts[i] = count
                
                print(f"\n类别{c}的错误预测分布:")
                for pred_class, count in pred_counts.items():
                    percentage = count / (total - correct) * 100
                    print(f"  预测为类别{pred_class}: {count}个样本 ({percentage:.1f}%)")
        
        # 使用t-SNE降维可视化
        visualize_comparison(combined_embed, data.y, global_pred)

# 可视化嵌入和预测结果
print("\n生成可视化结果...")
visualize_embeddings()

# 保存融合模型预测结果
result_file = "results/pubmed_expert_fusion_results.txt"
with open(result_file, 'w') as f:
    f.write(f"训练集准确率: {train_acc:.4f}\n")
    f.write(f"验证集准确率: {val_acc:.4f}\n")
    f.write(f"测试集准确率: {test_acc:.4f}\n\n")
    
    f.write("各类别准确率详情:\n")
    f.write("--------------------------------------------\n")
    f.write("| 类别  | 准确率  | 样本数量 | 正确预测数 |\n")
    f.write("--------------------------------------------\n")
    for i, acc in enumerate(test_class_accs):
        # 计算正确预测的样本数
        correct_count = int(acc * test_counts[i])
        f.write(f"| 类别{i} | {acc:.4f} | {test_counts[i]:<8} | {correct_count:<10} |\n")
    f.write("--------------------------------------------\n\n")
    
    f.write("\n融合策略:\n")
    f.write("使用自适应置信度加权融合策略:\n")
    f.write("1. 类别0: 主要由GNN模型负责 (基础权重0.95)，辅以attn2模型\n")
    f.write("2. 类别1: 主要由GNN模型负责 (基础权重0.95)，辅以attn2模型\n")
    f.write("3. 类别2: 主要由attn2模型负责 (基础权重0.8)，辅以GNN模型\n")
    f.write("4. 所有预测都通过模型置信度动态调整权重，表现更好的模型获得更高权重\n")
    f.write("\n模型权重路径:\n")
    f.write(f"GNN模型: results/best_gnn_model.pt\n")
    f.write(f"attn2模型: results/best_attn_model_class2.pt\n")

print(f"\n实验结果已保存至 {result_file}")

# 另外保存一个详细的预测结果，用于后续分析
pred_file = "results/pubmed_expert_fusion_predictions.pt"
torch.save({
    'global_pred': global_pred,
    'global_probs': global_probs,
    'true_labels': data.y,
    'test_mask': data.test_mask,
    'val_mask': data.val_mask,
    'train_mask': data.train_mask,
    'test_acc': test_acc,
    'class_accs': test_class_accs,
    'class_counts': test_counts
}, pred_file)

print(f"预测结果已保存至 {pred_file}")

# 输出结束信息
print("\nPubMed专家融合模型评估完成!") 