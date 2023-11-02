import torch
from torch_geometric.loader import DataLoader
import os
import pickle
from sklearn.model_selection import KFold
import numpy as np
import random
from torch_geometric.nn import global_mean_pool, GATConv
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import datetime

print("...............data loading...............")

# 定义数据集
data_path = "/home/bli/homology/Alphafold_test/colab_train_2031"

dataset = []  # data数据对象的list集合
for filename in os.listdir(data_path):
    file_path = os.path.join(data_path, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f).to(torch.device('cuda'))
    dataset.append(data)

# 设置随机数种子
seed = 2024
torch.manual_seed(seed) 
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# 设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 1300  # 输入特征的维度
hidden_channels = 512  # 隐层特征的维度
num_classes = 1  # 分类类别的数量
# num_heads = 7  # 注意力头的数量

# 打乱数据集的顺序
random.shuffle(dataset)

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
            if output.dim() == 0:
                output = output.unsqueeze(0)
            y_hat = torch.cat((y_hat, output),0) 
            y_true = torch.cat((y_true, data.y),0)
    return y_hat, y_true

# 五折交叉验证
kfold = KFold(n_splits=5)

print("...............data loading completed...............")
print("...............5 kFold train started...............")

batch_size = 16

# 存储每个 num_heads 的 r² 值
r2_values = []

# 循环遍历 num_heads
for num_heads in range(19, 21):
    k = 0   #计数第几折
    r2_per_num_heads = []   # 存储每个 num_heads 的五折交叉验证结果
    for train_idx, test_idx in kfold.split(dataset):
        k += 1
        print("...............num_heads = " + str(num_heads) + " : " + str(k) + "/5 kFold is training...............")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset,
                                sampler=train_subsampler,
                                batch_size=batch_size)
        test_loader = DataLoader(dataset,
                            sampler=test_subsampler,
                            batch_size=batch_size)
        
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
                torch.save(model.state_dict(), '/home/bli/homology/Alphafold_test/kFold_' + str(k) + "_best_model.pt")
            # print(f'Epoch: {epoch}, Train_Loss: {train_accuracy:.8f}, Test_Loss: {test_accuracy:.8f}')

            # print('Training finished.')


        model.load_state_dict(torch.load('/home/bli/homology/Alphafold_test/kFold_' + str(k) + "_best_model.pt"))
        model.eval()
        test_loss = test(model, device, test_loader, criterion)

        y_hat, y_true = predictions (model, device, test_loader)

        from sklearn import metrics
        from scipy.stats import pearsonr

        r2 = metrics.r2_score(y_true.cpu(), y_hat.cpu())
        pearson = pearsonr(y_true.cpu(), y_hat.cpu())
        # print(f'test loss: {test_loss:.8f}, R2: {r2:.8f}, Pearson: {pearson[0]:.8f}')
        y_hat = y_hat.cpu().numpy()
        y_true = y_true.cpu().numpy()

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
        
        # 获取当前时间
        current_time = datetime.datetime.now()

        # 格式化当前时间为字符串
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # 计算 r² 值并添加到 r2_per_num_heads 列表中
        r2_per_num_heads.append(r2)

        # 打印当前时间
        print(f'{current_time_str} R2: {r2:.3f}, test loss: {test_loss:.3f}, Pearson: {pearson[0]:.3f}, Accuracy: {binary_acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}, MCC: {mcc:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}')
    
        del model
        torch.cuda.empty_cache()  # 清空GPU缓存（如果在GPU上运行）
    # 计算 r² 值的中值和标准偏差
    median_r2 = sum(r2_per_num_heads) / len(r2_per_num_heads)
    std_r2 = (max(r2_per_num_heads) - min(r2_per_num_heads))/2

    # 将中值和标准偏差显示出来
    print(f'For num_heads = {num_heads}, Median R² = {median_r2:.3f} ± {std_r2:.3f}')
print("...............5 kFold train ended...............")