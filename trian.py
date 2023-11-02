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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("data loading...............")
train_dataset = [] # data数据对象的list集合
test_dataset = [] # data数据对象的list集合
val_dataset = [] # data数据对象的list集合

train_path = "/home/bli/GNN/Graph_bin/data/homology/alphafold_test/fold_completed_pkl_2697/train"
test_path = "/home/bli/GNN/Graph_bin/data/homology/alphafold_test/fold_completed_pkl_2697/test"
val_path = "/home/bli/Database/Scerevisiae/109/distance_10/109_pkl"

for filename in os.listdir(train_path):
  file_path = os.path.join(train_path, filename)
  with open(file_path, 'rb') as f:
    data = pickle.load(f).to(torch.device('cuda'))
  train_dataset.append(data)

for filename in os.listdir(test_path):
  file_path = os.path.join(test_path, filename)
  with open(file_path, 'rb') as f:
    data = pickle.load(f).to(torch.device('cuda'))
  test_dataset.append(data)

for filename in os.listdir(val_path):
  file_path = os.path.join(val_path, filename)
  with open(file_path, 'rb') as f:
    data = pickle.load(f).to(torch.device('cuda'))
  val_dataset.append(data)

# 设置随机数种子
seed = 2024
torch.manual_seed(seed) 
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# 打乱数据集的顺序
random.shuffle(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# print("data loaded !!!!!!!!!!")

# 定义图神经网络模型
class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.lin = nn.Linear(hidden_channels * num_heads, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Global pooling to obtain a fixed-size representation
        x = self.lin(x)
        return x.squeeze()  # 压缩输出维度为1

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
            y_hat = torch.cat((y_hat, output), 0) 
            y_true = torch.cat((y_true, data.y), 0)
    return y_hat, y_true


# 设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 1300  # 输入特征的维度
hidden_channels = 512  # 隐层特征的维度
num_classes = 1  # 分类类别的数量
num_heads = 16  # 注意力头的数量

# 创建模型实例
model = GATClassifier(in_channels, hidden_channels, num_heads).to(device)

#初始化参数
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)

# 定义损失函数和优化器
initial_lr = 0.000005 # 学习率
epochs = 15  # 训练轮数
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
    val_accuracy = test(model, device, val_loader, criterion)
    if test_accuracy < best_loss:
        best_loss = test_accuracy
        torch.save(model.state_dict(), '/home/bli/GNN/Graph_bin/data/homology/alphafold_test/best_model.pt')
    print(f'Epoch: {epoch}, Train_Loss: {train_accuracy:.8f}, Test_Loss: {test_accuracy:.8f}, ValLoss: {val_accuracy:.8f}')

print('Training finished.')


model.load_state_dict(torch.load("/home/bli/GNN/Graph_bin/data/homology/alphafold_test/best_model.pt"))
model.eval()
test_loss = test(model, device, test_loader, criterion)
val_loss = test(model, device, val_loader, criterion)

y_hat, y_true = predictions(model, device, test_loader)
val_hat, val_true = predictions(model, device, val_loader)

from sklearn import metrics
from scipy.stats import pearsonr

r2 = metrics.r2_score(y_true.cpu(), y_hat.cpu())
r2_val = metrics.r2_score(val_true.cpu(), val_hat.cpu())
pearson = pearsonr(y_true.cpu(), y_hat.cpu())
pearson_val = pearsonr(val_true.cpu(), val_hat.cpu())
print(f'test loss: {test_loss:.8f}, R2: {r2:.8f}, Pearson: {pearson[0]:.8f}')
print(f'val loss: {val_loss:.8f}, R2_val: {r2_val:.8f}, Pearson_val: {pearson_val[0]:.8f}')

y_hat = y_hat.cpu().numpy()
y_true = y_true.cpu().numpy()
val_hat = val_hat.cpu().numpy()
val_true = val_true.cpu().numpy()


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
binary_evaluate(val_true, val_hat, cut_off = 0.5)