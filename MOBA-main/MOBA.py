import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from loss import *
import numpy as np
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
import math
from torch.nn import Parameter

from torch_geometric.utils import dropout_adj, degree, to_undirected, get_laplacian
from torch_geometric.nn import GCNConv, GATConv
import scipy.stats as stats
from utils import *




class NewNewEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GATConv, k: int = 2, skip=False):
        super(NewNewEncoder, self).__init__()
        self.base_model = base_model
        # self.adj = adj
        assert k >= 1
        self.k = k  # 2
        self.skip = skip
        self.out = out_channels  # 512
        #self.dropout = nn.Dropout(0.5)
        hi = 1#1#2#3#6  #3
        if k == 1:
            self.conv = [base_model(in_channels,
                                    out_channels).jittable()]  # .jittable(): 这部分是调用 jittable() 方法。在 PyTorch 中，jittable() 通常表示启用了即时编译（JIT，Just-In-Time Compilation）的版本，即一种在运行时对代码进行优化的技术。
            self.conv = nn.ModuleList(self.conv)
            self.activation = creat_activation_layer(activation)
        elif not self.skip:
            self.conv = [base_model(in_channels, hi * out_channels)]  # 新的卷积层，[1433,1*512] #amac是767，256*2
            for _ in range(1, k - 1):
                self.conv.append(base_model(hi * out_channels, hi* out_channels))
                #self.conv.append(base_model(hi * out_channels, hi * out_channels))
            self.conv.append(base_model(hi * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = creat_activation_layer(activation)
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(
                self.conv)  # nn.ModuleList(...): 这是 PyTorch 提供的模块列表容器。它接受一个可迭代对象（通常是包含 PyTorch 模块的列表）作为参数，并将其转换为模块列表。
            # 这样的做法通常用于在神经网络中组织多个层或模块，使得它们能够方便地被管理和调用。在神经网络中，nn.ModuleList 可以用于存储卷积层、线性层或其他类型的层，以便在整个网络的前向传播中使用。这种方式也使得模型的结构更加灵活，可以动态添加或删除层。
            self.activation = creat_activation_layer(activation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, R=[1, 2], final=False):
        if final == False:
            K2 = np.random.randint(0, R[1])
            K1 = np.random.randint(0, R[0])
            #K2=R[1]
            #K1=R[0]
        if final:
            K1 = 1
            K2 = 1
        feat = x
        #feat = self.dropout(feat)
        #edge_index_1 = dropout_adj(edge_index, p=0.5)[0]
        #edge_index_2 = dropout_adj(edge_index, p=0.5)[0]
        #x = (self.conv[0](feat, edge_index_1, K1))
        x = (self.conv[0](feat, edge_index, K1))
        #x = self.activation(self.conv[0](feat, edge_index, K1))  # 用的GAT
        #x = self.dropout(x)
        #x = self.conv[1](x, edge_index_2, K2)
        x = self.conv[1](x, edge_index, K2)
        #x = self.activation(x)
        return x



class NewNewGRACE(torch.nn.Module):
    def __init__(self, encoder: NewNewEncoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(NewNewGRACE, self).__init__()
        self.encoder: NewNewEncoder = encoder
        #self.encoder2: Encoder = encoder2
        #self.BCE = torch.nn.BCELoss()
        self.tau: float = tau #0.4
        #self.adj = ADJ #0
        #self.adj = adj
        #self.A = A
        #self.norm = (A.shape[0] * A.shape[0]) / (float((A.shape[0] * A.shape[0] - torch.sum(A))) * 2)
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden) #512->512
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden) #512->512

        self.num_hidden = num_hidden #512

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, R = [1,2], final = False) -> torch.Tensor:
        return self.encoder(x, edge_index, R, final)#, self.encoder2(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        #z = self.fc1(z)
        return self.fc2(z)



class NewGConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(NewGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels#输入通道数1433，也就是X的shape[1]
        self.out_channels = out_channels#输出通道数512
        self.improved = improved#$设置为true时A尖等于A+2I

        self.cached = cached#If set to True, the layer will cache the computation of D^−1/2A^D^−1/2 on first execution, and will use the cached version for further executions. This parameter should only be set to True in transductive learning scenarios. (default: False)
        self.normalize = normalize#是否添加自环并应用对称归一化。

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:#如果设置为False，则该层将不会学习加法偏差
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)#glorot函数下面有写，初始化weight矩阵
        zeros(self.bias)#zeros函数下面有写，初始化偏置矩阵
        self.cached_result = None
        self.cached_num_edges = None


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes) #原来的边索引加上2708 #fill_value: 如果图中某些节点没有自环，可以通过 fill_value 参数指定自环的默认权重。如果设置为 None，则默认使用 1.0。
 #add_remaining_self_loops 函数的目的是确保图中包含了所有可能的自环，并在缺失的自环位置上添加自环。返回的 edge_index 和 edge_weight 是更新后的边索引和权重。
        row, col = edge_index #边索引直接分配
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) #dim_size=num_nodes 表示相加后的张量在第一个维度上的大小为 num_nodes。这样，deg 张量的每个元素 deg[i] 表示节点 i 的度（即与节点 i 相连的边的权重之和）。这在图神经网络中是一种常见的操作，用于计算节点的度信息。
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight


    def forward(self, x, edge_index, c, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)#将x与权重矩阵相乘变成[2708，512]

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(
                    self.node_dim), edge_weight, self.improved, x.dtype) #用度来对边进行加权 #self.node_dim看一下是多少，是不是节点的数量
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        #x_1 = x
        if c == 0:
            #x = self.propagate(edge_index, x=x)
            x = self.propagate(edge_index, x=x, norm=norm)
        for _ in range(c):
           # x = 1 * x + 1 * self.propagate(edge_index, x=x)

            x = x + 1 * self.propagate(edge_index, x=x, norm=norm) #在 PyTorch Geometric 中，propagate 函数是 GNN（Graph Neural Network）中消息传播的核心函数。它用于在图结构上传播节点之间的消息，其中 edge_index 表示图的边的索引，x 表示节点的特征，norm 表示用于规范化消息传播的权重。
            #在 GNN 中，消息传播是通过邻接矩阵（通常由 edge_index 表示）进行的。propagate 函数根据图的结构，将节点的特征信息在图中传递，更新节点的表示。
            x =  x/2
        return x


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

def glorot(tensor):#inits.py中
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)#将tensor的值设置为-stdv, stdv之间
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)





def edgeidx2sparse(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    else:
        raise ValueError(name)
    return layer


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    if activation == "elu":
        return nn.ELU()
    if activation == "relu":
        return nn.ReLU()
    if activation == "prelu":
        return nn.PReLU()
    else:
        raise ValueError("Unknown activation")

class GNNEncoder(nn.Module):
    def __init__(
            self,
            in_channels, #1433
            hidden_channels, #256
            out_channels, #128
            num_layers=2, #2
            dropout=0.5, #0.4
            bn=False,
            layer="gcn",
            activation="elu",
            use_node_feats=True,
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity #nn.BatchNorm1d 是 PyTorch 中用于一维数据（通常用于神经网络中的全连接层）的批归一化层。批归一化是一种用于加速深度神经网络训练的技术，它有助于防止梯度消失或梯度爆炸，并提高模型的稳定性和泛化性能
        self.use_node_feats = use_node_feats

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(bn(second_channels * heads))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x) #2708*1433
            x = conv(x, edge_index) #gcn的卷积2708*256
            x = self.bns[i](x) #nn.Identity 是 PyTorch 中的一个内置模块，用于表示恒等映射。它是 torch.nn 模块中的一个类，用于在神经网络中引入恒等函数。使用 nn.Identity 可以创建一个什么都不做的层，即输入与输出相同。这在构建神经网络时可以用于实现跳跃连接或者在某些情况下进行模型组合。
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index) #[2708,128]
        x = self.bns[-1](x)
        x = self.activation(x)
        return x




class Con_Projector(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()

        self.proj = nn.Linear(in_channels, in_channels) #128
    def forward(self, x):
        x = self.proj(x)
        return x

class Projector(nn.Module):
    """Simple MLP Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels #128 2. 256
            second_channels = out_channels if i == num_layers - 1 else hidden_channels #1. 256  2.1433
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, x):
        for i, mlp in enumerate(self.mlps[:-1]): #[128,256]   [256,1433]
            x = self.dropout(x) #2708*128
            x = mlp(x) #2708*256
            x = self.activation(x)
        x = self.dropout(x)   
        x = self.mlps[-1](x) #2708*1433
        x = self.activation(x)  
        return x



def random_negative_sampler(edge_index, num_nodes, num_neg_samples): #这行代码的目的是生成一个形状为 (2, num_neg_samples) 的张量，其中的元素是在 [0, num_nodes) 范围内随机整数。
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index) #然后，通过 .to(edge_index) 将生成的张量转换到与 edge_index 张量相同的设备（例如 CPU 或 GPU）。这可能用于负采样或其他类似的任务，其中需要在节点集合中随机选择一些节点。
    return neg_edges

class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList() #nn.ModuleList 是 PyTorch 中的一个容器，用于包装多个 nn.Module 对象。它允许在神经网络中动态地添加、管理和调用多个子模块。

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, z_1, z_2, edge, sigmoid=True, reduction=False):
        x = z_1[edge[0]] * z_2[edge[1]] #edge[2,8400]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]): #全链接层mlp会执行梯度下降吗
            x = self.dropout(x) #8400*128
            x = mlp(x) #[8400,64]
            x = self.activation(x)
        x = self.mlps[-1](x) #[8400,1]

        if sigmoid:
            return x.sigmoid()
        else:
            return x

    
class Feature_learner(nn.Module):
    def __init__(self, x,adj,node, feature,vali_test_id,train_id):
        super(Feature_learner, self).__init__()
        self.x = x
        self.adj= adj
        self.node = node
        self.feature = feature
        self.feature_init = torch.eye(self.node).cuda()
        self.vali_test_id = vali_test_id
        self.train_id =train_id
        a1=[]
        a2=[]

        adj = self.adj.to_dense()
        #tmp_mean_2 = torch.ones(1, self.feature).cuda()
        #tmp_mean = torch.ones(1, self.feature).cuda()
        #tmp_mean = self.x[2268,:].cuda()
        #tmp_mean_2 = self.x[2268, :].cuda()

        for i in self.vali_test_id:
            tmpdata = torch.where(adj[i, :] != 0)[0]
            tmpdata_2=torch.where(adj[tmpdata, :] != 0)[1]#用1没问题
            #tmpdata_3 = torch.where(adj[tmpdata_2, :] != 0)[1]
            a1 = []
            tmp_mean = torch.zeros(1, self.feature)
            for q in range(len(tmpdata)):
                if tmpdata[q] in self.train_id:
                    a1.append(tmpdata[q])
            if (len(a1) > 0):
                tmp_mean = torch.zeros(1, self.feature)
                a3 = torch.tensor(self.x[a1, :])
                #for w in range(self.feature):
                tmp_mean = torch.mean(a3,dim=0)
                #self.x[i, :] = tmp_mean/(q+1)
                #self.x[i, :] = tmp_mean
            #elif(len(a1)==0):
            a2 = []
            tmp_mean_2 = torch.zeros(1, self.feature)
            for e in range(len(tmpdata_2)):
                if tmpdata_2[e] in self.train_id:
                    a2.append(tmpdata_2[e])
            if (len(a2)>0):
                tmp_mean_2 = torch.zeros(1, self.feature)
                a4 = torch.tensor(self.x[a2, :])
                #for w in range(self.feature):
                tmp_mean_2 = torch.mean(a4,dim=0)
            '''
            无用
            a3 = []
            #tmp_mean_3 = torch.zeros(1, self.feature).cuda()
            for e in range(len(tmpdata_3)):
                if tmpdata_3[e] in self.train_id:
                    a3.append(tmpdata_3[e])
            if (len(a3) > 0):
                tmp_mean_3 = torch.zeros(1, self.feature).cuda()
                a4 = torch.tensor(self.x[a3, :])
                for w in range(self.feature):
                    tmp_mean_3[:, w] = torch.mean(a4[:, w])
            '''

            if tmp_mean!=None and tmp_mean_2!=None:
                self.x[i, :] = 4*tmp_mean+4*tmp_mean_2#+tmp_mean_3#2,1,1#2,4
            elif tmp_mean!=None:
                self.x[i, :] = 2 * tmp_mean
            elif tmp_mean_2!=None:
                self.x[i, :] = 4*tmp_mean_2
            else: continue

        '''
        for i in self.train_id:
            tmpdata = torch.where(adj[i, :] != 0)[0]
            
            for q in range(len(tmpdata)):
                if tmpdata[q] in self.vali_test_id:
                    a1.append(tmpdata[q])
                    a2.append(self.x[i,:])
        for k in self.vali_test_id:
            a1 = torch.tensor(a1)
            a2 = torch.from_numpy(np.stack(a2 ,axis=0))
            temp =torch.where(a1==k)[0]
            if len(temp)>0:#可以考虑二阶的用std
                a3 = torch.tensor(a2[temp,:])
                tmp_mean = torch.zeros(1, self.feature).cuda()
                for w in range(self.feature):
                    tmp_mean[:,w]= torch.mean(a3[:,w])
                self.x[k,:] = tmp_mean
            '''
        #self.fc = nn.Parameter(torch.zeros(self.node, self.feature))
        #self.fc = nn.Parameter(torch.randn(self.node, self.feature))
        #self.fc = nn.Parameter(2*torch.randn((self.node, self.feature))) #recall好而mlp和gcn差
        self.fc = nn.Parameter(1.0*torch.Tensor(self.x)+1*torch.randn((self.node, self.feature)))
        #self.fc = nn.Parameter(1.0 * torch.Tensor(self.x) + 1.0 * torch.ones(self.node, self.feature))
        #self.fc =(1.0*torch.Tensor(self.x)+0.2*torch.randn((self.node, self.feature))).cuda()
        #self.fc =torch.Tensor(self.x).cuda()
        #self.fc = torch.zeros(self.node, self.feature).cuda()
    def forward(self, x):
        z = torch.mm(self.feature_init, self.fc) #z[2708,1433]
        return z

'''
class Feature_learner_1(nn.Module):
    def __init__(self,diff, x,adj,node, feature,vali_test_id,train_id):
        super(Feature_learner_1, self).__init__()
        self.diff = diff
        self.x = x
        self.adj= adj
        self.node = node
        self.feature = feature
        self.feature_init = torch.eye(self.node).cuda()
        self.vali_test_id = vali_test_id
        self.train_id =train_id
        adj = self.adj.to_dense()
        for i in self.vali_test_id:
            tmpdata = torch.where(adj[i, :] != 0)[0]
            tmpdata_2 = torch.where(adj[tmpdata, :] != 0)[1]
            #tmpdata_3 = torch.where(adj[tmpdata_2, :] != 0)[1]
            a1 = []
            #tmp_mean = torch.zeros(1, self.feature).cuda()
            for q in range(len(tmpdata)):
                if tmpdata[q] in self.train_id:
                    a1.append(tmpdata[q])
            if (len(a1) > 0):
                tmp_mean = torch.zeros(1, self.feature).cuda()
                a4 = torch.tensor(self.x[a1, :])
                for w in range(self.feature):
                    tmp_mean[:, w] = torch.mean(a4[:, w])
                self.x[i, :] = tmp_mean
        #self.fc = nn.Parameter(torch.randn((self.node, self.feature)))
        self.fc = nn.Parameter(2*torch.Tensor(self.x) + 0.3* torch.randn((self.node, self.feature)))
        #self.fc = nn.Parameter(torch.Tensor(self.x))
    def forward(self, x):
        z = torch.mm(self.feature_init, self.fc) #z[2708,1433]
        return z
'''

def sample_negative_index(negative_number=0, epoch=0, epochs=0, Number=0):
    lamda = 1 / 2
    lower, upper = 0, Number - 1  # =512
    mu = ((epoch) / epochs) ** lamda * (
                upper - lower)  # 我们设置一个自步函数作为学习过程中采样的上界。我们使用随机抽样的简单样本来进行学习。该方法可以增强网络学习的鲁棒性。此外，对于每个时代，我们不断更新上界的大小，使用动态学习机制来学习不同的信息。#mu一点点增加，这就是简单到复杂的课程学习吧
    X = stats.uniform(1, mu)
    index = X.rvs(negative_number)
    index = index.astype(np.int64)
    return index

'''
def sample_node_negative_index(negative_number=0, epoch=0, epochs=0, Number=0):
    lamda = 1 / 2
    lower, upper = 0, Number - 1  # 防止取到第2708个点
    mu = ((epoch) / epochs) ** lamda * (upper - lower)
    X = stats.uniform(1, mu)
    index = X.rvs(negative_number)

    mu_1 = ((epoch - 1) / epochs) ** lamda * (upper - lower)
    if epoch > 10:
        mu_1 = ((epoch - 10) / epochs) ** lamda * (upper - lower)
    mu_2 = ((epoch) / epochs) ** lamda * (upper - lower)
    X = stats.uniform(mu_1, mu_2 - mu_1)  # 对于节点水平上的均匀随机抽样，采样间隔被设置为[g（t−c），g (t)]。
    index = X.rvs(negative_number)  # 随机采样样本数量设置为k，最后使用采样后得到的负样本进行训练。
    index = index.astype(np.int64)
    return index


def Sim_feat_loss_selfpace(Z, temperature=1.0, epoch=0, epochs=0, s=10):
    N = Z.shape[0]  # S一拔^n,
    index = sample_negative_index(negative_number=s, epoch=epoch, epochs=epochs, Number=N)  # 抽样（·）是指基于课程学习的统一随机抽样函数
    index = torch.tensor(index).cuda()

    sim_matrix = torch.exp(torch.pow(Z,
                                     2) / temperature)  # （S一拔^n）^2,抽样（·）是指基于课程学习的统一随机抽样函数。这种机制促进中心节点保持离相邻节点的距离，远离节点级的其他节点，从而使学习到的嵌入更具鉴别性。
    positive_samples_ii_jj = torch.diag(sim_matrix).reshape(N, 1)  # 对于特征相似度矩阵，我们提出了一个特征相似度损失函数，它将对角线上的值作为正样本，其他值作为负样本
    positive_samples = torch.sum(positive_samples_ii_jj, 1)

    sim_matrix_sort, _ = torch.sort(sim_matrix, dim=0,
                                    descending=False)  # sim_matrix_sort 包含了对 sim_matrix 张量每列进行升序排序后的结果。这样的操作在很多场景中都是有用的，例如在图神经网络中选择邻居节点、在推荐系统中选择相似物品等。
    negative_samples = sim_matrix_sort.index_select(0,
                                                    index)  # 0是第0维，最终，negative_samples 包含了根据 index 中的行索引从 sim_matrix_sort 中选择的行。这样的操作通常在负采样（negative sampling）的场景中使用，例如在训练推荐系统或图神经网络时，从排序好的相似性矩阵中选择负样本。
    negative_samples = torch.sum(negative_samples, 0)

    loss = (- torch.log(positive_samples / negative_samples)).mean()

    return loss

'''
def Sim_feat_loss_selfpace(Z , temperature = 1.0, epoch = 0, epochs = 0,s=10,b=0):
    '''
    N = Z.shape[0] #S一拔^n,
    index = sample_negative_index(negative_number=s, epoch=epoch, epochs=epochs, Number=N) #抽样（·）是指基于课程学习的统一随机抽样函数
    index = torch.tensor(index).cuda()
    aa = [1]
    a1= torch.LongTensor(aa).cuda()
    #sim_matrix = Z #/2708 #（S一拔^n）^2,抽样（·）是指基于课程学习的统一随机抽样函数。这种机制促进中心节点保持离相邻节点的距离，远离节点级的其他节点，从而使学习到的嵌入更具鉴别性。
    positive_samples_ii_jj = torch.diag(sim_matrix).reshape(N, 1) #对于特征相似度矩阵，我们提出了一个特征相似度损失函数，它将对角线上的值作为正样本，其他值作为负样本
    positive_samples = torch.sum(positive_samples_ii_jj,1)
    #sim_matrix_sort = torch.argsort(-sim_matrix, axis=0)
    sim_matrix_sort, _ = torch.sort(sim_matrix, dim=0, descending=False) #sim_matrix_sort 包含了对 sim_matrix 张量每列进行升序排序后的结果。这样的操作在很多场景中都是有用的，例如在图神经网络中选择邻居节点、在推荐系统中选择相似物品等。
    negative_samples = sim_matrix_sort.index_select(0, index) #0是第0维，最终，negative_samples 包含了根据 index 中的行索引从 sim_matrix_sort 中选择的行。这样的操作通常在负采样（negative sampling）的场景中使用，例如在训练推荐系统或图神经网络时，从排序好的相似性矩阵中选择负样本。
    negative_samples = torch.sum(negative_samples, 0)/len(index)
    negative_samples =torch.exp(negative_samples)
    #negative_samples = negative_samples.sum()

    '''
    off_diagonal_mask = ~torch.eye(Z.shape[1]).bool().cuda()
    #loss = ( -torch.log(positive_samples / negative_samples)).mean()
    sim_matrix = Z.cuda()
    a2 = sim_matrix.diagonal().cuda()
    sim_matrix = sim_matrix[off_diagonal_mask]
    sim_matrix = sim_matrix.pow(2)
    #sim_matrix_sort, _ = torch.sort(sim_matrix, dim=0,descending=True)
    #a1=torch.tensor(range(0,1600)).cuda()
    #negative_samples = sim_matrix_sort.index_select(0,a1)
    #loss =((torch.exp((1 -a2).pow(2)).sum()+(torch.exp(sim_matrix).sum())/1433)/500).cuda()
    #loss = ((torch.exp((1 - a2).pow(2)).sum() + (torch.exp(sim_matrix).sum()) / Z.shape[1]/10) / Z.shape[1]).cuda()
    loss =((1 -a2).mean()+sim_matrix.mean()).cuda()
    #loss = ((torch.exp((1 - a2).pow(2)).sum() + (torch.exp(negative_samples).sum()) / 1433) / 500).cuda()
    #print(loss)

    return loss

'''
def Sim_node_loss_selfpace(Z, adj_label12, temperature=1.0, epoch=0, epochs=0, s=10):
    N = Z.shape[0]
    index = sample_node_negative_index(negative_number=s, epoch=epoch, epochs=epochs,
                                       Number=N)  # 抽样（·）是指基于课程学习的统一随机抽样函数
    index = torch.tensor(index).cuda()
    sim_matrix = torch.exp(torch.pow(Z, 2) / temperature)  # 抽样（·）是指基于课程学习的统一随机抽样函数。这种机制促进中心节点保持离相邻节点的距离，远离节点级的其他节点
    positive_samples = torch.sum(sim_matrix * adj_label12, 1)
    sim_matrix_sort, _ = torch.sort(sim_matrix, dim=0, descending=False)  # 因此，该网络将专注于从得分较低的样本中学习
    negative_samples = sim_matrix_sort.index_select(0, index)

    negative_samples = torch.sum(negative_samples, 0)
    loss = (- torch.log(positive_samples / negative_samples)).mean()
    return loss
'''

class Model(nn.Module):
    def __init__(
            self,
            #diff,
            #encoder,
            edge_decoder,
            projector,
            con_projector,
            temp,
            pos_weight_tensor, neg_weight_tensor,
            MAmodel,
            mask=None, #
            random_negative_sampling=False,
            loss="ce",

    ):
        super().__init__()
        #self.diff = diff
        #self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.projector = projector
        self.con_projector = con_projector
        self.mask = mask
        self.temp = temp
        self.pos_weight_tensor = pos_weight_tensor
        self.neg_weight_tensor = neg_weight_tensor
        self.MAmodel = MAmodel
        if loss == "ce":
            self.loss_edgefn = ce_loss
        else:
            raise ValueError(loss)
        self.contrastive_loss = calc_loss
        self.rec_loss = fts_rec_loss

        if random_negative_sampling:
            self.negative_sampler = random_negative_sampler
        else:
            self.negative_sampler = negative_sampling

    '''
    def nor(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value,
            num_nodes)  # 原来的边索引加上2708 #fill_value: 如果图中某些节点没有自环，可以通过 fill_value 参数指定自环的默认权重。如果设置为 None，则默认使用 1.0。
        # add_remaining_self_loops 函数的目的是确保图中包含了所有可能的自环，并在缺失的自环位置上添加自环。返回的 edge_index 和 edge_weight 是更新后的边索引和权重。
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0,
                          dim_size=num_nodes)  # dim_size=num_nodes 表示相加后的张量在第一个维度上的大小为 num_nodes。这样，deg 张量的每个元素 deg[i] 表示节点 i 的度（即与节点 i 相连的边的权重之和）。这在图神经网络中是一种常见的操作，用于计算节点的度信息。
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight
        '''


    def forward(self, data_1, data_2, norm_adj, feature_learner,train_fts_idx, vali_test_fts_idx): #这个好像不执行梯度下降

        x_1_, edge_index_1 = data_1.x, data_1.edge_index

        x_learn = feature_learner(x_1_)
        zero_ = torch.zeros_like(x_learn, device=x_learn.device)
        zero = torch.zeros_like(x_learn, device=x_learn.device)
        zero[vali_test_fts_idx] = zero_[vali_test_fts_idx] + x_learn[vali_test_fts_idx]
        #x_1__ =  x_learn.#x_1__ =  x_1_
        x_1__ = zero + x_1_

        x_1 = torch.mm(norm_adj, x_1__)
        x_2, edge_index_2 = data_2.x, data_2.edge_index #在这之前和train_one_epoch都是一样的
        #zero[vali_test_fts_idx] = zero_[vali_test_fts_idx] + x_2[vali_test_fts_idx]
        #x_2 = zero + x_1_
        #x_2 =feature_learn_1(x_2) +x_2 # torch.randn(x_2.size()[0],x_2.size()[1],device=x_2.device)
        #x_2 =torch.mm(self.diff, x_2)
        #z_1 = self.MAmodel(x_1, edge_index_1, [2, 2], True)#原来是False
        #z_2 = self.MAmodel(x_2, edge_index_1, [8, 8], True)
        z_1 = self.MAmodel(x_1, edge_index_1, [2, 2], False)
        z_2 = self.MAmodel(x_2, edge_index_1, [10, 10], False)
        ##z_1 = self.MAmodel(x_1, edge_index_1, [2, 2], True)
        ##z_2 = self.MAmodel(x_2, edge_index_1, [2, 2], True)
        #z_1 = self.encoder(x_1, edge_index_1)
        #z_2 = self.encoder(x_2, edge_index_1)
        z = (z_1 + z_2) * 0.5 #本来是到278*128的，但是project又转成2708*1433
        #z=z_1
        out = self.projector(z)
        return out






    def train_one_epoch(
            self, data_1, data_2,
            #data_1_1,data_2_2,
            norm_adj, feature_learner,train_fts_idx, vali_test_fts_idx,epoch,epochs,adj, batch_size=2 ** 16):

        x_1_, edge_index_1 = data_1.x, data_1.edge_index

        #x_1_1, edge_index_1_1 = data_1_1.x, data_1_1.edge_index

        x_learn = feature_learner(x_1_) #x_learn与x_1_没关系，是随机呈正态分布的的2708*1433的数
        zero_ = torch.zeros_like(x_learn, device=x_learn.device)
        zero = torch.zeros_like(x_learn, device=x_learn.device)
        zero[vali_test_fts_idx] = zero_[vali_test_fts_idx] + x_learn[vali_test_fts_idx] #zero就是x_learn，之前把验证测试集的属性删除了，现在又加上了随机正态分布的属性
        x_1__ =  zero + x_1_
        #x_1__ =  x_1_#x_1__ = x_learn
        x_1 = torch.mm(norm_adj, x_1__)
        x_2, edge_index_2 = data_2.x, data_2.edge_index #一样
        #zero[vali_test_fts_idx] = zero_[vali_test_fts_idx] + x_2[vali_test_fts_idx]
        #x_2 = zero + x_1_
        #x_2 =feature_learn_1(x_2)+x_2    #torch.randn(x_2.size()[0], x_2.size()[1], device=x_2.device)
        #x_2 = torch.mm(self.diff, x_2)
        #x_2 = feature_learn_1(x_2) +x_2
        #x_2_2, edge_index_2_2 = data_2_2.x, data_2_2.edge_index
        remaining_edges, masked_edges = self.mask(edge_index_1)
        #remaining_edges_1 ,masked_edges_1 = self.mask_1(edge_index_1)
        #remaining_edges_1, masked_edges_1 = self.mask(edge_index_1_1)

        aug_edge_index, _ = add_self_loops(edge_index_1) #在 PyTorch Geometric 中，图通常由两个张量表示：一个包含边的源节点索引和目标节点索引，以及一个包含边的特征的张量。当调用 add_self_loops 时，它会在图中为每个节点添加一个自环，从而改变图的结构。
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=data_1.num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges) #tensor(2,8400)这一步通过 view 操作将张量重塑为一个新的形状，其中第一个维度的大小为 2，而第二个维度的大小根据原始张量的总大小自动确定。-1 表示 PyTorch 应该根据其他维度的大小自动计算该维度的大小


        '''
        def nor(edge_index, num_nodes, edge_weight=None, improved=False,
                dtype=None):
            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                         device=edge_index.device)

            fill_value = 1 if not improved else 2
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value,
                num_nodes)  # 原来的边索引加上2708 #fill_value: 如果图中某些节点没有自环，可以通过 fill_value 参数指定自环的默认权重。如果设置为 None，则默认使用 1.0。
            # add_remaining_self_loops 函数的目的是确保图中包含了所有可能的自环，并在缺失的自环位置上添加自环。返回的 edge_index 和 edge_weight 是更新后的边索引和权重。
            row, col = edge_index
            deg = scatter_add(edge_weight, row, dim=0,
                              dim_size=num_nodes)  # dim_size=num_nodes 表示相加后的张量在第一个维度上的大小为 num_nodes。这样，deg 张量的每个元素 deg[i] 表示节点 i 的度（即与节点 i 相连的边的权重之和）。这在图神经网络中是一种常见的操作，用于计算节点的度信息。
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

            return edge_index, edge_weight
        '''


        self.node_dim=-2
        edge_weight = None
        self.improved = False
        #idx = np.random.permutation(x_1[0])
        #seq1 = x_1[idx, :]
       # remaining_edges_1, norm = nor(remaining_edges, x_2.size(self.node_dim), edge_weight, self.improved, x_2.dtype)

        for perm in DataLoader(
                range(masked_edges.size(1)), batch_size=batch_size, shuffle=True
        ): #8433个数打乱
            #z_1 = self.MAmodel(x_1, edge_index_1, [2, 2],True)#原来是False
            #z_2 = self.MAmodel(x_2, edge_index_1, [10, 10],True)
            #node_deg = degree(edge_index_1[1])
            #feature_weights = feature_drop_weights_dense(x_1, node_c=node_deg).cuda()
            #x_1 = drop_feature_weighted_2(x_1, feature_weights,0.01)
            #x_2 = drop_feature_weighted_2(x_1, feature_weights,0.01)
            z_1 = self.MAmodel(x_1, remaining_edges, [2, 2], False)
            z_2 = self.MAmodel(x_2, remaining_edges, [10, 10], False)#remaining_edges
            #z_1 = self.encoder(x_1, remaining_edges) #[2708,1433]->[2708,128]
            #z_1_1 = self.encoder(x_1, remaining_edges_1.cuda())
            #aa = MessagePassing()
            #z_2= 1 * x_2 + 1 *aa.propagate(remaining_edges_1,x = x_2,norm = norm)
            #z_2 = 0.5 * z_2
            #z_2 = 1 * z_2 + 1 * aa.propagate(remaining_edges_1, x=z_2, norm=norm)
            #x_2 = 0.5 * z_2
            #z_2 = self.encoder(x_2, remaining_edges)

            batch_masked_edges  = masked_edges[:, perm] #相当于按列打乱edge_index_1,masked_edges
            batch_neg_edges = neg_edges[:, perm]  #是这个
            #batch_neg_edges_1 = neg_edges_1[:, perm]

            pos_out_1 = self.edge_decoder(
                z_1, z_2, batch_masked_edges, sigmoid=False
            ) #得到[8400:1]
            neg_out_1 = self.edge_decoder(z_1, z_2, batch_neg_edges, sigmoid=False)

            pos_out_2 = self.edge_decoder(
                z_2, z_1, batch_masked_edges, sigmoid=False
            )
            neg_out_2 = self.edge_decoder(z_2, z_1, batch_neg_edges, sigmoid=False)

            loss_edge = self.loss_edgefn(pos_out_1,neg_out_1) + self.loss_edgefn(pos_out_2,neg_out_2) #ce_loss对应公式10，11



            z_1_p = z_1
            z_2_p = z_2
            #adj = adj.to_dense().cuda()

            feat = torch.mm(F.normalize((z_1_p+z_2_p ).T), F.normalize((z_1_p +z_2_p).T).T)  # 其为S^n计算特征级余弦相似度 ，2708*512变成512*512 # [512,512] #F.normalize就是每一行的数除以L2范数（每个数平方求和后开根号），使数据都处于零到一之间，距离就有上界了，样本间的差异就小了
            feat_loss = Sim_feat_loss_selfpace(feat, temperature=1.0, epoch=epoch, epochs=epochs, s=50,b=vali_test_fts_idx[0:270])
            node = torch.mm(F.normalize(z_1_p+z_2_p ), F.normalize(z_1_p+z_2_p).T)  # [2708,2708]
            node_loss = Sim_feat_loss_selfpace(node, temperature=10.0, epoch=epoch, epochs=epochs, s=50,b=vali_test_fts_idx[0:270])
            # node_loss = Sim_node_loss_selfpace(node, adj, temperature=10.0, epoch=epoch, epochs=epochs, s=50)

            '''
            n_data, d = z_1_p.shape
            similarity = torch.matmul(z_1_p, torch.transpose(z_1_p, 1,
                                                               0))  # detach() 方法用于创建一个新的张量，其值与调用 detach() 方法的张量相同，但不再随着计算图的变化而变化，即将其从计算图中分离出来。这样做的目的通常是为了避免梯度的传播，从而保持张量的值不变。
            similarity += torch.eye(n_data, device=0) * 10  # 对角线元素＋10

            _, I_knn = similarity.topk(k=4, dim=1, largest=True,
                                       sorted=True)  # 在嵌入向量中，对每个节点中4个最可能是同一标签的节点(包括本身） #用于计算相似度矩阵中每个节点的 top-k 最相似的节点索引.topk() 是一个 PyTorch 提供的方法，用于在指定维度上获取最大或最小的 k 个值及其对应的索引。在这个例子中，topk() 方法被调用在 dim=1 的维度上，也就是在每行上寻找最相似的节点。
            knn_neighbor = create_sparse(I_knn)  # 得到每个节点语义最相近的4个点共46804个点，并化为图级稀疏矩阵
            locality = knn_neighbor * adj.cuda()
            ind = locality.coalesce()._indices()
            '''

            #z_1_p = self.projector(z_1)
            #z_2_p = self.projector(z_2)
            loss1 = loss_fn(z_1_p[edge_index_1[0]], z_2_p[edge_index_1[1]])
            #loss2 = loss_fn(z_1_p[batch_masked_edges[0]], z_2_p[batch_masked_edges[1]]) + loss_fn(z_2_p[batch_masked_edges[0]], z_1_p[batch_masked_edges[1]])
            #loss3 = loss_fn_n(z_1_p[batch_neg_edges[0]], z_2_p[batch_neg_edges[1]])
            #loss4 = loss_fn_n(z_1_p[batch_neg_edges[1]], z_2_p[batch_neg_edges[0]])
            #loss5 = loss_fn_n(z_1_p[batch_neg_edges_1[1]], z_2_p[batch_neg_edges_1[0]])
            #loss6 = loss_fn_n(z_1_p[batch_neg_edges_1[1]], z_2_p[batch_neg_edges_1[0]])
            loss =  loss1.mean() #+ loss3.mean()+ loss4.mean()
            #loss = loss1.mean() + loss3.mean()


            '''
            z1 = (z_1 - z_1.mean(0)) / z_1.std(0)  # h1.mean(0)为512个数, z1结果仍为2708,512
            z2 = (z_2 - z_2.mean(0)) / z_2.std(0)
            c   = torch.mm(z1.T, z2)
            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)  # 都是512*512
            # loss_1 = (z1 - z2).pow(2).sum()
            N = 2708
            c = c / N
            c1 = c1 / N
            c2 = c2 / N
            loss_inv = -torch.diagonal(c).sum()
            iden = torch.tensor(np.eye(c.shape[0])).cuda()
            loss_dec1 = (iden - c1).pow(2).sum()  # 512*512内全部数都求和
            loss_dec2 = (iden - c2).pow(2).sum()

            loss_wu = loss_inv + 0.001 * (loss_dec1 + loss_dec2)
            '''
            loss_con = self.contrastive_loss(z_1_p, z_2_p, temperature=self.temp) #cal_loss 对应公式8，9
            z = (z_1 + z_2) * 0.5
            #z = z_1


            x_recon = self.projector(z) #2708*128得2708*1433

            loss_recon = self.rec_loss(x_recon[train_fts_idx], x_1_[train_fts_idx], self.pos_weight_tensor,self.neg_weight_tensor)

            #输出与自设标签的对比（0/1）
            #loss_recon = self.rec_loss(x_recon[train_fts_idx], x_1_[train_fts_idx], self.pos_weight_tensor,self.neg_weight_tensor)
            #loss_total = loss_edge +loss_con + loss_recon#+ 3*feat_loss
            #loss_total = loss_edge + feat_loss +0.001*node_loss+ loss_recon
            #loss_total =  0.15*loss +loss_recon+1*feat_loss #+ 1*node_loss#+loss_ncl
            loss_total = 1*loss_edge + 0.2*loss + 1*loss_recon + 1*node_loss +0.1*feat_loss  #+ loss_con#+ 0.1*loss，5feature,0.1node    0.15 1 1 1;0.1， 0.2 1 1 0.1
            #if epoch % 10 == 0:
                #print(loss)
            #loss_total =0.1*loss_wu + loss_recon+feat_loss
        return loss_total





