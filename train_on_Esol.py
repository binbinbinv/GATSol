import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")
import pickle
from tqdm import tqdm
from torch_geometric.data import DataLoader
import random
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv
import torch.nn.functional as F

# 定义图神经网络模型
class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, num_layers):
        super(GATClassifier, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads))
            else:
                self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads))
        self.lin = nn.Linear(hidden_channels * num_heads, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x.squeeze()

# 定义训练函数
def train(model, device, loader, optimizer, criterion):
    model.train()
    
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.float(), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader.dataset)    
    return avg_loss


# 定义测试函数
def test(model, device, loader, criterion):
    model.eval()
    loss = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss += criterion(output, data.y).item()
    return loss/len(loader.dataset)

def predictions(model, device, loader):
    model.eval()
    y_hat = torch.tensor([]).cuda()
    y_true = torch.tensor([]).cuda()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            if output.dim() == 0:
                output = output.unsqueeze(0)
            y_hat = torch.cat((y_hat, output),0) 
            y_true = torch.cat((y_true, data.y),0)
    return y_hat, y_true

def load_graph_dataset(data_path):
    dataset = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f).to(torch.device('cuda'))
        dataset.append(data)
    return dataset

def binary_evaluate(y_true, y_hat, cut_off = 0.5):
  binary_pred = [1 if pred >= cut_off else 0 for pred in y_hat]
  binary_true = [1 if true >= cut_off else 0 for true in y_true]
  binary_acc = metrics.accuracy_score(binary_true, binary_pred)
  precision = metrics.precision_score(binary_true, binary_pred)
  recall = metrics.recall_score(binary_true, binary_pred)
  f1 = metrics.f1_score(binary_true, binary_pred)
  auc = metrics.roc_auc_score(binary_true, y_hat)
  mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
  TN, FP, FN, TP = metrics.confusion_matrix(binary_true, binary_pred).ravel()
  sensitivity = 1.0 * TP / (TP + FN)
  specificity = 1.0 * TN / (FP + TN)
  print(f'Accuracy: {binary_acc:.8f}, Precision: {precision:.8f}, Recall: {recall:.8f}, F1: {f1:.8f}, AUC: {auc:.8f}, MCC: {mcc:.8f}, Sensitivity: {sensitivity:.8f}, Specificity: {specificity:.8f}')

# 设置随机数种子
seed = 2024
torch.manual_seed(seed) 
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

print("data loading...............")

train_path = "/home/bli/GNN/Graph_bin/data/homology/alphafold_test/fold_completed_pkl_BLOSUM62+ESM/train"
test_path = "/home/bli/GNN/Graph_bin/data/homology/alphafold_test/fold_completed_pkl_BLOSUM62+ESM/test"

train_dataset = load_graph_dataset(train_path)
test_dataset = load_graph_dataset(test_path)

# 打乱数据集的顺序
random.shuffle(train_dataset)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
# print("data loaded !!!!!!!!!!")

# 设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 1300  # 输入特征的维度
hidden_channels = 1024  # 隐层特征的维度
num_classes = 1  # 分类类别的数量
num_heads = 16  # 注意力头的数量
num_layers = 2  # 网络层数

# 创建模型实例
model = GATClassifier(in_channels, hidden_channels, num_heads, num_layers).to(device)

#初始化参数
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)

# 定义损失函数和优化器
initial_lr = 0.000001 # 学习率
epochs = 20  # 训练轮数
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=initial_lr)

#开始训练和测试
best_loss = float('inf')  # 初始最佳损失设为无穷大

# print('seed -- ' + str(seed)+' -- Training start...............')
model.train()
for epoch in range(1, epochs + 1):
    if epoch < 10:
        lr = initial_lr / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, device, train_loader, optimizer, criterion)
    train_accuracy = test(model, device, train_loader, criterion)
    test_accuracy = test(model, device, test_loader, criterion)
    if test_accuracy < best_loss:
        best_loss = test_accuracy
        torch.save(model.state_dict(), '/home/bli/GNN/Graph_bin/data/homology/best_model.pt')
    print(f'Epoch: {epoch}, Train_Loss: {train_accuracy:.8f}, Test_Loss: {test_accuracy:.8f}')

model.load_state_dict(torch.load("/home/bli/GNN/Graph_bin/data/homology/best_model.pt"))
model.eval()
test_loss = test(model, device, test_loader, criterion)

y_hat, y_true = predictions(model, device, test_loader)

from sklearn import metrics
from scipy.stats import pearsonr

r2 = metrics.r2_score(y_true.cpu(), y_hat.cpu())
pearson = pearsonr(y_true.cpu(), y_hat.cpu())
print(f'test loss: {test_loss:.8f}, R2: {r2:.8f}, Pearson: {pearson[0]:.8f}')

y_hat = y_hat.cpu().numpy()
y_true = y_true.cpu().numpy()

def binary_evaluate(y_true, y_hat, cut_off = 0.5):
  binary_pred = [1 if pred >= cut_off else 0 for pred in y_hat]
  binary_true = [1 if true >= cut_off else 0 for true in y_true]
  binary_acc = metrics.accuracy_score(binary_true, binary_pred)
  precision = metrics.precision_score(binary_true, binary_pred)
  recall = metrics.recall_score(binary_true, binary_pred)
  f1 = metrics.f1_score(binary_true, binary_pred)
  auc = metrics.roc_auc_score(binary_true, y_hat)
  mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
  TN, FP, FN, TP = metrics.confusion_matrix(binary_true, binary_pred).ravel()
  sensitivity = 1.0 * TP / (TP + FN)
  specificity = 1.0 * TN / (FP + TN)
  print(f'Accuracy: {binary_acc:.8f}, Precision: {precision:.8f}, Recall: {recall:.8f}, F1: {f1:.8f}, AUC: {auc:.8f}, MCC: {mcc:.8f}, Sensitivity: {sensitivity:.8f}, Specificity: {specificity:.8f}')

binary_evaluate(y_true, y_hat, cut_off = 0.5)