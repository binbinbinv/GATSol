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
test_dataset = [] # data数据对象的list集合

test_path = "/home/bli/homology/dataset/Esol/fold_completed_pkl_BLOSUM62+ESM/test"

for filename in os.listdir(test_path):
  file_path = os.path.join(test_path, filename)
  with open(file_path, 'rb') as f:
    data = pickle.load(f).to(torch.device('cuda'))
  test_dataset.append(data)


batch_size = 4
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
print("data loaded !!!!!!!!!!")

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
        self.lin1 = nn.Linear(hidden_channels * num_heads, 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        # x = self.lin1(x)
        return x.squeeze()

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


# 设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 1300  # 输入特征的维度
hidden_channels = 1024  # 隐层特征的维度
num_classes = 1  # 分类类别的数量
num_heads = 16  # 注意力头的数量
num_layers = 2  # 网络层数

# 创建模型实例
model = GATClassifier(in_channels, hidden_channels, num_heads, num_layers).to(device)

model.load_state_dict(torch.load("/home/bli/homology/best_model.pt"))
model.eval()

# 定义损失函数和优化器
criterion = nn.MSELoss(reduction='sum')

#开始测试
test_loss = test(model, device, test_loader, criterion)

y_hat, y_true = predictions(model, device, test_loader)

from sklearn import metrics
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve

y_hat = y_hat.cpu()
y_true = y_true.cpu()

binary_pred = [1 if pred >= 0.5 else 0 for pred in y_hat]
binary_true = [1 if true >= 0.5 else 0 for true in y_true]

# 输出测试集ROC曲线横纵坐标
fpr, tpr, thresholds = roc_curve(binary_true, y_hat)
df = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
df.to_csv('/home/bli/homology/Roc.csv', index=False)

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

