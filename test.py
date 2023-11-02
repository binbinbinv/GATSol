import os
import pandas as pd
import iFeatureOmegaCLI
import torch
import numpy as np
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")
import pickle
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import DataLoader
import random
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR

print("val_data loading...............")
val_dataset = [] # data数据对象的list集合

for filename in os.listdir("/home/bli/GNN/Graph_bin/data/homology/val_pkl"):
  file_path = os.path.join("/home/bli/GNN/Graph_bin/data/homology/val_pkl", filename)
  with open(file_path, 'rb') as f:
    data = pickle.load(f).to(torch.device('cuda'))
  val_dataset.append(data)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
print("val_data loaded !!!!!!!!!!")

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

def precision(model, device, loader):
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
num_heads = 7  # 注意力头的数量

# 创建模型实例
model = GATClassifier(in_channels, hidden_channels, num_heads).to(device)

model.load_state_dict(torch.load("/home/bli/GNN/Graph_bin/data/homology/alphafold_test/best_model.pt"))
model.eval()

criterion = nn.MSELoss(reduction='sum')
test_loss = test(model, device, val_loader, criterion)

y_hat, y_true = precision(model, device, val_loader)

from sklearn import metrics
from scipy.stats import pearsonr

r2 = metrics.r2_score(y_true.cpu(), y_hat.cpu())
pearson = pearsonr(y_true.cpu(), y_hat.cpu())
print(f'test loss: {test_loss:.8f}, R2: {r2:.8f}, Pearson: {pearson[0]:.8f}')
y_hat = y_hat.cpu().numpy()
y_true = y_true.cpu().numpy()

df_hat_true = pd.DataFrame({'y_hat': y_hat, 'y_true': y_true})
df_hat_true.to_csv("hat_true.csv", index=False)

binary_pred = [1 if pred >= 0.5 else 0 for pred in y_hat]
binary_true = [1 if true >= 0.5 else 0 for true in y_true]

# binary evaluate
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
