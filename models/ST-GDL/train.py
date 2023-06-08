import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from torch.utils.data import Dataset, DataLoader
from model import STDGL
import pickle

# 读取数据
data_path = './data_pro/'
graph_path = './data_ori/'
print('loading data...')
# data = np.load(data_path + 'data.npz')
# print(data.files)
# xc = data['closeness']
# xt = data['trend']
# xp = data['period']
# label = data['prediction']
xc = np.load(data_path + 'data_closeness.npy')
print(xc.shape)
xt = np.load(data_path + 'data_trend.npy')
print(xt.shape)
xp = np.load(data_path + 'data_period.npy')
label = np.load(data_path + 'data_prediction.npy')

# 读取eoGraph.dgl
# geoGraph = dgl.load_graphs(graph_path + 'geoGraph.dgl')[0][0]
# adj = geoGraph.adjacency_matrix()
with open(graph_path+'geoGraph.pkl', 'rb') as f:
    adj_ori = pickle.load(f)
print('loading data finished')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 将数据转换为tensor
xc = torch.from_numpy(xc).float()
xt = torch.from_numpy(xt).float()
xp = torch.from_numpy(xp).float()
label = torch.from_numpy(label).float()
# adj = torch.from_numpy(adj)

# 定义dataset
class MyDataset(Dataset):
    def __init__(self, xc, xt, xp, label):
        self.xc = xc
        self.xt = xt
        self.xp = xp
        self.label = label
    def __getitem__(self, index):
        return self.xc[index], self.xt[index], self.xp[index], self.label[index]
    def __len__(self):
        return len(self.xc)

# 定义dataloder
train_dataset = MyDataset(xc, xt, xp, label)
#分割数据集
train_size = int(0.8 * len(train_dataset))
val_size = int(0.1 * len(train_dataset))
test_size = len(train_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

# 定义模型
model = STDGL(in_channels = 361, out_channels = 361)
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 定义损失函数
criterion = nn.MSELoss()

# 定义early stopping 满足条件时停止训练并保存模型
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    # 如果val_loss下降则保存模型
    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # 如果val_loss没有下降则计数器+1
        elif score > self.best_score - self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            # 如果计数器达到patience则停止训练
            if self.counter >= self.patience:
                self.early_stop = True
        # 如果val_loss下降则保存模型
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    # 保存模型
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.best_score, val_loss))
        torch.save(model.state_dict(), './checkpoint.pt')
        self.best_score = val_loss

early_stopping = EarlyStopping(patience=10, verbose=True)

# 训练
def train(model, device, train_loader, optimizer, epoch):
    # 将模型放入device
    model.to(device)
    model.train()
    adj = [item.to(device) for item in adj_ori]
    for idx, (xc, xt, xp, label) in enumerate(train_loader):
        # xc, xt, xp, label = xc.to(device), xt.to(device), xp.to(device), label.to(device)
        xc = xc.to(device)
        xt = xt.to(device)
        xp = xp.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(xc, xt, xp, adj)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        # 每10个batch打印一次loss
        batch_idx = idx + 1
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(xc), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# 验证
def val(model, device, val_loader):
    model.to(device)
    model.eval()
    val_loss = 0
    adj = [item.to(device) for item in adj_ori]
    with torch.no_grad():
        for xc, xt, xp, label in val_loader:
            xc, xt, xp, label = xc.to(device), xt.to(device), xp.to(device), label.to(device)
            output = model(xc, xt, xp, adj)
            val_loss += criterion(output, label).item()
    val_loss /= len(val_loader)
    print('Val set: Average loss: {:.4f}'.format(val_loss))
    # 满足条件时停止训练并保存模型
    early_stopping(val_loss, model)
            
# 测试
def test(model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    adj = [item.to(device) for item in adj_ori]
    with torch.no_grad():
        for xc, xt, xp, label in test_loader:
            xc, xt, xp, label = xc.to(device), xt.to(device), xp.to(device), label.to(device)
            # 将列表放入devevice中
            output = model(xc, xt, xp, adj)
            test_loss += criterion(output, label).item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))

# 训练模型
print('start training...')
for epoch in range(1, 10):
    train(model, device, train_loader, optimizer, epoch)
    val(model, device, val_loader)
test(model, device, val_loader)






