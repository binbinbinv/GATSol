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

def name_seq_dict(path):
    pdb_chain_list = pd.read_csv(path, header=0)
    dict_pdb_chain = pdb_chain_list.set_index('id')['sequence'].to_dict()
    return dict_pdb_chain

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

def predictions(model, device, loader):
    model.eval()
    y_hat = torch.tensor([]).cuda()
    y_true = torch.tensor([]).cuda()
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            output = model(data)
            if output.dim() == 0:
                output = output.unsqueeze(0)
            y_hat = torch.cat((y_hat, output),0) 
            y_true = torch.cat((y_true, data.y),0)
    return y_hat, y_true

def print_box(message):
    box_width = 40
    message = f" {message} "
    padding = (box_width - len(message)) // 2
    border = '*' * box_width
    padding_str = '*' + ' ' * padding

    print(border)
    print(padding_str + message + ' ' * (box_width - len(padding_str) - len(message)) + '*')
    print(border)

# 在框框中间显示 "Prediction begin"
print_box("Prediction Begin")

pkl_path = "./NEED_to_PREPARE/pkl"
name_dict = name_seq_dict("./NEED_to_PREPARE/list.csv")
file_names = list(name_dict.keys())

test_dataset = [] # data数据对象的list集合

for filename in file_names:
  file_path = os.path.join(pkl_path, filename+".pkl")
  with open(file_path, 'rb') as f:
    data = pickle.load(f).to(torch.device('cuda'))
  test_dataset.append(data)


batch_size = 1
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

# 设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 1300  # 输入特征的维度
hidden_channels = 1024  # 隐层特征的维度
num_classes = 1  # 分类类别的数量
num_heads = 16  # 注意力头的数量
num_layers = 2  # 网络层数

# 创建模型实例
model = GATClassifier(in_channels, hidden_channels, num_heads, num_layers).to(device)

model.load_state_dict(torch.load("../check_point/best_model/best_model.pt"))
model.eval()

y_hat, y_true = predictions(model, device, test_loader)

y_hat = list(y_hat.cpu().numpy())

df = pd.read_csv("./NEED_to_PREPARE/list.csv")

df["Solubility_hat"] = y_hat

# 保存修改后的 DataFrame 到 CSV 文件
df.to_csv("./Output.csv", index=False)

print_box("Prediction Completed and Check the Output.csv")
