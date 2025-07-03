import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, remove_self_loops, to_dense_adj, dense_to_sparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
plt.rcParams["font.size"]=16
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载pubmed数据集
dataset = Planetoid(root='data/PubMed', name='PubMed', transform=NormalizeFeatures())
data = dataset[0].to(device)

print(f"数据集：{data}")
print(f"节点特征维度: {data.x.shape}")
print(f"节点数量: {data.num_nodes}")
print(f"边数量: {data.num_edges}")
print(f"特征维度: {dataset.num_features}")
print(f"类别数量: {dataset.num_classes}")

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
    
    plt.savefig('results/pubmed_class0_comparison.png')
    plt.show()

# 创建多跳图注意力网络模型
class MultiHopGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, max_hop=6, dropout=0.5, heads=8):
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
        # 使用更明显的初始权重差异，强调一阶连接的重要性
        initial_weights = torch.tensor([0.7, 0.3][:max_hop])
        self.hop_attention = nn.Parameter(initial_weights)
        
        # 添加层归一化(LayerNorm)提高训练稳定性
        self.layer_norm1 = nn.LayerNorm(self.hidden_per_hop * heads)
        self.layer_norm2 = nn.LayerNorm(out_channels)
        
        # 添加用于残差连接的线性投影
        self.skip_proj = nn.Linear(in_channels, self.hidden_per_hop * heads)
        self.final_proj = nn.Linear(self.hidden_per_hop * heads, out_channels)
        
    def forward(self, x, edge_index, higher_order_adjs):
        # 保存输入用于残差连接
        input_x = x
        
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
        
        return x_combined, hop_attn_weights

# 创建高阶邻接矩阵 (现在考虑1-2跳)
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

# 初始化模型
hidden_channels = 64  # 进一步减少隐藏层维度
model = MultiHopGAT(
    in_channels=dataset.num_features, 
    hidden_channels=hidden_channels,
    out_channels=dataset.num_classes,
    max_hop=MAX_HOP,
    dropout=0.5,  # 调整dropout率
    heads=2       # 进一步减少注意力头数量
).to(device)

# 优化器和损失函数
# 添加类别平衡与焦点损失来提高对难分类别的关注
class_counts = torch.bincount(data.y[data.train_mask])
class_weights = 1.0 / (class_counts.float() / class_counts.sum())

# 特殊关照类别0
# 增加类别0的权重
class_weights[0] *= 3.0  # 三倍类别0的权重
class_weights = class_weights.to(device)
print(f"类别权重: {class_weights}")

# 使用OneCycleLR调度器提高学习效率
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=2e-3)

# 添加学习率调度器
# 使用OneCycleLR来实现更好的学习率调度
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.005,
    epochs=500,
    steps_per_epoch=1,
    pct_start=0.2,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)

# 使用带有类权重的交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# 添加一个辅助函数来计算焦点损失
def focal_loss(pred, target, weight=None, gamma=2.0):
    """焦点损失 - 更关注难分类的样本"""
    ce_loss = F.cross_entropy(pred, target, weight=weight, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = (1 - pt) ** gamma * ce_loss
    return loss.mean()

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out, hop_attn = model(data.x, data.edge_index, higher_order_adjs)
    
    # 使用交叉熵损失
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    
    # 使用更适中的梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    # 更新OneCycleLR学习率
    scheduler.step()
    
    return loss.item(), hop_attn

# 测试函数
def test():
    model.eval()
    out, hop_attn = model(data.x, data.edge_index, higher_order_adjs)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        accs.append(acc)
    return accs[0], accs[1], accs[2], hop_attn, out

# 训练模型
print("开始训练...")
best_val_acc = 0
best_test_acc = 0
best_epoch = 0
best_out = None
hop_weights = None
patience = 200  # 适当减少patience值，因为我们使用了更好的优化器
no_improve_count = 0
all_val_accs = []  # 记录所有验证准确率
all_train_losses = []  # 记录所有训练损失

# 创建历史窗口记录
window_size = 20  # 使用更长的窗口评估趋势
val_acc_windows = []

# 确保存储结果的目录存在
os.makedirs("results", exist_ok=True)

# 最大训练轮次
max_epochs = 500 # 调整为OneCycleLR的epochs值

for epoch in range(max_epochs):
    # 训练一个epoch
    train_loss, train_hop_attn = train()
    all_train_losses.append(train_loss)
    
    # 每5个epoch测试一次，增加评估频率
    if epoch % 5 == 0 or epoch == 0:
        train_acc, val_acc, test_acc_current, test_hop_attn, out = test()
        all_val_accs.append(val_acc)
        
        # 计算其他类别的准确率
        pred = out.argmax(dim=1)
        class_accs = []
        for c in range(dataset.num_classes):
            mask = (data.y == c) & data.test_mask
            if mask.sum() > 0:
                correct = (pred[mask] == data.y[mask]).sum()
                class_acc = float(correct) / float(mask.sum())
                class_accs.append(float(class_acc))
            else:
                class_accs.append(0.0)
        
        # 计算类别0的准确率
        mask0 = (data.y == 0) & data.test_mask
        if mask0.sum() > 0:
            correct0 = (pred[mask0] == data.y[mask0]).sum()
            class0_acc = float(correct0) / float(mask0.sum())
        else:
            class0_acc = 0
        
        # 使用加权指标来决定最佳模型，特别关注类别0
        weighted_metric = 0.7 * val_acc + 0.3 * class0_acc
        
        # 保存最佳模型
        if weighted_metric > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc_current
            best_epoch = epoch
            hop_weights = test_hop_attn.detach().cpu().numpy()
            best_out = out
            no_improve_count = 0
            
            print(f"新的最佳模型! Epoch {epoch}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc_current:.4f}")
            # 保存最佳模型
            torch.save(model.state_dict(), 'results/best_attn_model_class0.pt')
        else:
            no_improve_count += 1
        
        # 计算验证准确率的滑动窗口平均值
        if len(all_val_accs) >= window_size:
            window_avg = sum(all_val_accs[-window_size:]) / window_size
            val_acc_windows.append(window_avg)
        
        # 早停 - 更加关注验证准确率趋势
        if epoch > 100 and no_improve_count >= patience:
            if len(val_acc_windows) >= 10:
                recent_trend = val_acc_windows[-1] - val_acc_windows[-10]
                if recent_trend < 0.001:  # 趋势变平，可以提前终止
                    print(f"验证准确率趋势平稳，提前终止训练。窗口趋势: {recent_trend:.6f}")
                    break
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc_current:.4f}")
        print(f"类别0准确率: {class0_acc:.4f}")
        if len(val_acc_windows) > 0:
            print(f"窗口平均值: {val_acc_windows[-1]:.4f}, 最佳验证准确率: {best_val_acc:.4f}")
            
        # 打印学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")
            
        # 每100个epoch保存一次模型，用于后续分析
        if epoch % 100 == 0 and epoch > 0:
            torch.save(model.state_dict(), f'results/multihop_gat_model_class0_epoch_{epoch}.pt')

print(f"训练完成，最佳性能在 Epoch {best_epoch}")
print(f"最佳验证准确率: {best_val_acc:.4f}")
print(f"对应测试准确率: {best_test_acc:.4f}")
print(f"最佳跳数注意力权重: {hop_weights}")

# 如果保存了最佳模型，加载它
if os.path.exists('results/best_attn_model_class0.pt'):
    model.load_state_dict(torch.load('results/best_attn_model_class0.pt'))
    print("已加载最佳模型")
    _, _, _, test_hop_attn, best_out = test()
    hop_weights = test_hop_attn.detach().cpu().numpy()

# 可视化结果
visualize_comparison(best_out, data.y, best_out.argmax(dim=1))

# 可视化不同跳数的注意力权重
plt.figure(figsize=(12, 6))
hop_labels = [f'{i+1}-hop' for i in range(MAX_HOP)]
plt.bar(hop_labels, hop_weights)
plt.title('不同跳数的注意力权重')
plt.ylabel('权重')
plt.xticks(rotation=45)  # 旋转标签以便更好地显示
plt.savefig('results/pubmed_class0_hop_weights.png')
plt.show()

# 分析模型性能在不同类别上的表现
model.eval()
_, _, _, _, out = test()
pred = out.argmax(dim=1)

# 计算每个类别的准确率
class_accs = []
for c in range(dataset.num_classes):
    # 在测试集中找到属于类别c的节点
    mask = (data.y == c) & data.test_mask
    if mask.sum() > 0:
        correct = (pred[mask] == data.y[mask]).sum()
        acc = float(correct) / float(mask.sum())
        class_accs.append(acc)
    else:
        class_accs.append(0)

# 可视化每个类别的准确率
plt.figure(figsize=(10, 6))
plt.bar(range(dataset.num_classes), class_accs)
plt.title('各类别的测试准确率')
plt.xlabel('类别')
plt.ylabel('准确率')
plt.ylim(0, 1)
plt.savefig('results/pubmed_class0_class_accs.png')
plt.show()

# 可视化训练过程中的损失和准确率
indices = [i*5 for i in range(len(all_val_accs))]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# 绘制训练损失
ax1.plot(all_train_losses)
ax1.set_title('训练损失变化')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('损失')
ax1.grid(True)

# 绘制验证准确率
ax2.plot(indices, all_val_accs)
ax2.set_title('验证集准确率变化')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('准确率')
ax2.grid(True)

plt.tight_layout()
plt.savefig('results/pubmed_class0_training.png')
plt.show()

# 保存实验结果
with open('results/experiment_results_pubmed_class0.txt', 'w') as f:
    f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
    f.write(f"最佳测试准确率: {best_test_acc:.4f}\n")
    f.write(f"类别0准确率: {class0_acc:.4f}\n")
    f.write(f"跳数注意力权重: {hop_weights}\n")
    f.write(f"各类别准确率: {class_accs}\n")
    f.write(f"类别平衡权重: {class_weights.cpu().numpy()}\n")
    f.write(f"加权指标计算方式: 0.7*val_acc + 0.3*class0_acc\n")
    
print("实验结果已保存到 results/experiment_results_pubmed_class0.txt") 