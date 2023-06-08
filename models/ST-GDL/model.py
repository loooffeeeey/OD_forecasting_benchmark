import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# 定义一个三维卷积网络
class Conv3D_BN(nn.Module):
    def __init__(self, in_channels, out_channels=12, kernel_size=(3,1,1), stride=1, padding=0):
        super(Conv3D_BN, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = self.bn(x).squeeze()
        x = self.relu(x)
        return x

# 定义时空卷积网络
class STConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,3), NO=361,ND=361, NL=19, NH=19, stride=1, padding=(0,1)):
        super(STConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.AvgPool = nn.AvgPool2d(kernel_size=(12, NO),stride=1)
        self.MaxPool = nn.MaxPool2d(kernel_size=(12, NO),stride=1)
        self.avg_linear1 = nn.Linear(ND, NL)
        self.avg_linear2 = nn.Linear(NL, ND)
        self.max_linear1 = nn.Linear(ND, NL)
        self.max_linear2 = nn.Linear(NL, ND)
        self.sigmoid1 = nn.Sigmoid()
        self.NL = NL
        self.NH = NH
        self.conv3d = nn.Conv3d(12, 12, kernel_size=(5, 5, 2), stride=1, padding=(2,2,0))
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x_in):
        # x_in: b*361*12*361
        x = self.conv(x_in)
        x = self.bn(x)
        x_ca = self.relu(x)
        x_cag = self.sigmoid1(self.avg_linear2(self.avg_linear1(self.AvgPool(x_ca).squeeze())).unsqueeze(2).unsqueeze(3) + self.max_linear2(self.max_linear1(self.MaxPool(x_ca).squeeze())).unsqueeze(2).unsqueeze(3))
        # x 与 x_cag 元素相乘
        x_ca = x_ca * x_cag
        # 将dim=1的维度进行取均值
        x_mean = torch.mean(x_ca, dim=1)
        x_mean = x_mean.reshape(x_mean.size(0), x_mean.size(1), self.NL, self.NH)
        # 将dim=1的维度进行取最大值
        x_max = torch.max(x_ca, dim=1)[0]
        x_max = x_max.reshape(x_max.size(0), x_max.size(1), self.NL, self.NH)
        # 将x_mean和x_max进行拼接
        x = torch.stack([x_mean, x_max], dim=-1)# b*12*19*19*2
        x = self.sigmoid2(self.conv3d(x).squeeze().view(x.size(0), x.size(1), x.size(2) * x.size(3))).unsqueeze(1)
        # 将第3维和第4维合并
        # 在dim=1处增加维度
        x = x * x_ca
        x = x + x_in
        return x

# 图卷机层
class SGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGCN, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    # 传入邻接矩阵
    def forward(self, x, edge_index):
        x = x.permute(0, 2, 1,3)
        x = self.conv(x, edge_index[0], edge_index[1])
        # edge_index


        x = x.permute(0, 2, 1,3)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MP_O(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(1,3), NO=361,ND=361, NL=19, NH=19, stride=1, padding=(0,1)):
        super(MP_O, self).__init__()
        self.stc1 = STConv(in_channels, out_channels,kernel_size, NO,ND, NL, NH, stride, padding)
        self.stc2 = STConv(out_channels, out_channels,kernel_size, NO,ND, NL, NH, stride, padding)
        self.stc3 = STConv(out_channels, out_channels,kernel_size, NO,ND, NL, NH, stride, padding)
        self.sgcn = SGCN(out_channels, out_channels)
        self.lnorm1 = nn.LayerNorm(out_channels)
        self.lnorm2 = nn.LayerNorm(out_channels)
        self.conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0)
    def forward(self, x,adj):
        x = x.permute(0, 3, 1, 2)
        x = self.stc1(x)
        x = self.sgcn(x,adj)
        x = self.stc2(x)
        x = self.lnorm1(x)
        x = self.stc3(x)
        x = self.lnorm2(x)
        x = self.conv2d(x)
        x = x.permute(0, 2, 3, 1)
        return x

class MP_D(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(1,3), NO=361,ND=361, NL=19, NH=19, stride=1, padding=(0,1)):
        super(MP_D, self).__init__()
        self.stc1 = STConv(in_channels, out_channels,kernel_size, NO,ND, NL, NH, stride, padding)
        self.stc2 = STConv(out_channels, out_channels,kernel_size, NO,ND, NL, NH, stride, padding)
        self.stc3 = STConv(out_channels, out_channels,kernel_size, NO,ND, NL, NH, stride, padding)
        self.sgcn = SGCN(out_channels, out_channels)
        self.lnorm1 = nn.LayerNorm(out_channels)
        self.lnorm2 = nn.LayerNorm(out_channels)
        self.conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0)
    def forward(self, x,adj):
        x = x.permute(0, 2, 1, 3)
        x = self.stc1(x)
        x = self.sgcn(x,adj)
        x = self.stc2(x)
        x = self.lnorm1(x)
        x = self.stc3(x)
        x = self.lnorm2(x)
        x = self.conv2d(x)
        x = x.permute(0, 2, 1, 3)
        return x
    
class MP(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(1,3), NO=361,ND=361, NL=19, NH=19, stride=1, padding=(0,1)):
        super(MP, self).__init__()
        self.mpo = MP_O(in_channels, out_channels,kernel_size, NO,ND, NL, NH, stride, padding)
        self.mpd = MP_D(in_channels, out_channels,kernel_size, NO,ND, NL, NH, stride, padding)
    def forward(self, x ,adj): 
        x_o = self.mpo(x, adj)
        x_d = self.mpd(x, adj)
        x = x_o + x_d
        return x

class STDGL(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=(1,3), NO=361,ND=361, NL=19, NH=19, stride=1, padding=(0,1)):
        super(STDGL, self).__init__()
        self.mp_trend = MP(in_channels, out_channels,kernel_size, NO,ND, NL, NH, stride, padding)
        self.mp_period = MP(in_channels, out_channels,kernel_size, NO,ND, NL, NH, stride, padding)
        self.close = Conv3D_BN(in_channels = 1, out_channels =12)
        self.mp_closeness = MP(in_channels, out_channels,kernel_size = (3,3), NO=361,ND=361, NL=19, NH=19, stride=1, padding=(1,1))
        self.wt = nn.Parameter(torch.FloatTensor(1))
        self.wp = nn.Parameter(torch.FloatTensor(1))
        self.wc = nn.Parameter(torch.FloatTensor(1))
        self.final_conv = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=1, padding=0)
    def forward(self, xc,xp,xt, adj):
        x_trend = self.mp_trend(xt, adj)
        x_period = self.mp_period(xp, adj)
        x_close = self.close(xc)
        x_closeness = self.mp_closeness(x_close, adj)
        # 加权求和，权重为可训练参数
        x = x_trend * self.wt + x_period * self.wp + x_closeness * self.wc
        #b*12*19*19
        x = self.final_conv(x)
        return x
    
# 越执行网络占显存越大 //中间变量未释放
# dataset 300（约12天） 对应5G输入数据
# 动态图GCN