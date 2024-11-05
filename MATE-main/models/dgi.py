import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse) #节点嵌入2708*512

        c = self.read(h_1, msk)  #mask为空 #1*512 #得到图级的归纳向量（summary vectors)，们寻求获取节点（即局部）表示，以捕获整个图的全局信息内容，由一个汇总向量s（c)表示
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2) #最大化局部互信息的作用

        return ret

    # Detach the return variablesh_1.detach() 是在 PyTorch 中使用的方法，它用于从计算图中分离（detach）一个张量。这个操作的目的是阻止梯度通过该张量反向传播，即停止对该张量的梯度计算。
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk) #msk为空

        return h_1.detach(), c.detach()

