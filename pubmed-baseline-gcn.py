import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

# 设置中文字体
plt.rcParams["font.size"]=16
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查 CUDA 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载pubmed数据集
dataset = Planetoid(root='data/PubMed', name='PubMed', transform=NormalizeFeatures())
data = dataset[0].to(device)  # 将图数据移动到 CUDA 设备

print("数据集信息:")
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"类别数: {dataset.num_classes}")
print(f"特征维度: {dataset.num_node_features}")

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
    
    plt.show()

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
 
    def forward(self, x, edge_index):
        # 第一层卷积 + ReLU
        x = F.relu(self.conv1(x, edge_index))
        # 第二层卷积
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
 
# 初始化模型并将其移动到 CUDA 设备
hidden_channels = 512  # 隐藏层维度
model = GCN(
    in_channels=dataset.num_node_features, 
    hidden_channels=hidden_channels, 
    out_channels=dataset.num_classes
).to(device)

# 优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # 前向传播
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 计算损失
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试函数 
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # 预测类别
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        accs.append(acc)
    
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
    
    return accs[0], accs[1], accs[2], out, class_accs

# 训练模型
print("开始训练...")
best_val_acc = 0
best_test_acc = 0
best_epoch = 0
patience = 300  # 增加patience值，减少早停敏感度
no_improve_count = 0
all_val_accs = []  # 记录所有验证准确率
all_train_losses = []  # 记录所有训练损失

# 创建历史窗口记录
window_size = 20  # 使用更长的窗口评估趋势
val_acc_windows = []

for epoch in range(3000):  # 增加最大epoch数
    loss = train()
    all_train_losses.append(loss)
    
    # 每20个epoch测试一次，减少计算开销
    if epoch % 10 == 0 or epoch == 0:
        train_acc, val_acc, test_acc, out, class_accs = test()
        all_val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            no_improve_count = 0
            
            # 保存最佳模型
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), 'results/best_gcn_model.pt')
            best_out = out
        else:
            no_improve_count += 1
    
    # 计算移动窗口平均值（每20个epoch）
    if epoch % 10 == 0 and len(all_val_accs) >= window_size:
        window_avg = sum(all_val_accs[-window_size:]) / window_size
        val_acc_windows.append(window_avg)
        
        # 早停 - 使用更宽松的条件 (仅在足够长的训练后才考虑早停)
        if epoch > 500 and no_improve_count >= patience:
            # 检查验证准确率趋势
            if len(val_acc_windows) >= 10:  # 至少需要10个窗口才评估趋势
                recent_trend = val_acc_windows[-1] - val_acc_windows[-10]
                # 如果最近趋势为负且窗口平均值低于最佳值的0.97
                if recent_trend <= 0 and window_avg < 0.97 * best_val_acc:
                    print(f"早停于epoch {epoch}，长时间无改进且趋势下降")
                    break
        
    # 打印训练进度
    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        print(f"类别准确率: 类别0: {class_accs[0]:.4f}, 类别1: {class_accs[1]:.4f}, 类别2: {class_accs[2]:.4f}")
        if len(val_acc_windows) > 0:
            print(f"窗口平均值: {val_acc_windows[-1]:.4f}, 最佳验证准确率: {best_val_acc:.4f}")
        
    # 每100个epoch保存一次模型，用于后续分析
    if epoch % 100 == 0 and epoch > 0:
        torch.save(model.state_dict(), f'results/gcn_model_epoch_{epoch}.pt')

print(f"训练完成，最佳性能在 Epoch {best_epoch}")
print(f"最佳验证准确率: {best_val_acc:.4f}")
print(f"对应测试准确率: {best_test_acc:.4f}")

# 如果保存了最佳模型，加载它
if os.path.exists('results/best_gcn_model.pt'):
    model.load_state_dict(torch.load('results/best_gcn_model.pt'))
    print("已加载最佳模型")
    _, _, _, best_out, final_class_accs = test()
    print(f"最终类别准确率: 类别0: {final_class_accs[0]:.4f}, 类别1: {final_class_accs[1]:.4f}, 类别2: {final_class_accs[2]:.4f}")
else:
    best_out = out  # 使用最后一个epoch的输出
    final_class_accs = class_accs

# 可视化模型结果
visualize_comparison(best_out, data.y, best_out.argmax(dim=1))

# 分析模型性能在不同类别上的表现
model.eval()
out = best_out  # 使用最佳模型的输出
pred = out.argmax(dim=1)

# 使用之前计算的最终类别准确率
class_accs = final_class_accs

# 计算每个类别在测试集中的样本数量
class_counts = []
for c in range(dataset.num_classes):
    mask = (data.y == c) & data.test_mask
    class_counts.append(int(mask.sum().item()))

print("\n各类别在测试集中的样本数量:")
for c in range(dataset.num_classes):
    print(f"类别 {c}: {class_counts[c]} 个样本")

# 可视化每个类别的准确率
plt.figure(figsize=(10, 6))
plt.bar(range(dataset.num_classes), class_accs)
plt.title('各类别的测试准确率')
plt.xlabel('类别')
plt.ylabel('准确率')
plt.ylim(0, 1)
plt.show()

# 可视化训练过程中的损失和准确率
indices = [i*10 for i in range(len(all_val_accs))]
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
plt.show()

# 保存实验结果
os.makedirs("results", exist_ok=True)
with open('results/experiment_results_gcn.txt', 'w') as f:
    f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
    f.write(f"最佳测试准确率: {best_test_acc:.4f}\n")
    f.write("各类别准确率:\n")
    for c in range(dataset.num_classes):
        f.write(f"类别 {c}: {class_accs[c]:.4f} (样本数: {class_counts[c]})\n")
    
print("实验结果已保存到 results/experiment_results_gcn.txt") 