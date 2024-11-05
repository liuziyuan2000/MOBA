import torch
import torch.nn.functional as F



def fts_rec_loss(recon_x=None, x=None, p_weight=None, n_weight=None):
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    output_fts_reshape = torch.reshape(recon_x, shape=[-1])#二维数据转成一维
    out_fts_lbls_reshape = torch.reshape(x, shape=[-1])
    weight_mask = torch.where(out_fts_lbls_reshape != 0.0, p_weight, n_weight) #torch.where 的作用是对于 out_fts_lbls_reshape 中不等于 0.0 的元素，选择对应位置的 p_weight 元素，对于等于 0.0 的元素，选择对应位置的 n_weight 元素，最终构建出一个新的张量 weight_mask。这种操作通常在损失函数的计算中用于引入样本权重，以便对不同样本或类别赋予不同的重要性
    loss_bce = torch.mean(BCE(output_fts_reshape, out_fts_lbls_reshape) * weight_mask)
    return loss_bce


def ce_loss(pos_out,neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out)) #正样例的当然是和标签为一的进行对比啊，通过交叉熵损失函数得到一个确定值的张量。#torch.ones_like 是 PyTorch 中的一个函数，用于创建一个与给定张量形状相同的张量，并且所有元素的值都设置为 1
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out)) #Sigmoid 激活函数,将输入映射到范围 [0, 1] 之间。 主要用于二分类问题，以产生一个表示概率的输出。在二分类问题中，通常在模型的最后一层使用 sigmoid 激活函数，将输出视为表示正类别概率的单一值。
    return pos_loss + neg_loss
 #F.binary_cross_entropy,二分类任务,输入模型的最后一层是一个单一的概率值（例如，使用 sigmoid 激活函数），表示正类别的概率。
def calc_loss(x, x_aug, temperature=2.0, sym=True):
    batch_size = x.shape[0]
    x_abs = x.norm(dim=1)#L2范数，数的平方累加求和再开根号
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (torch.einsum('i,j->ij', x_abs, x_aug_abs) + 1e-8) #就是ik点乘kj得到ij #用einsum函数好像必须行列维度一样。 equation: 定义了爱因斯坦求和的方程式，例如 "ij,jk->ik"。方程式中的字母表示张量的维度，箭头 "->" 后面表示输出张量的维度。

    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if sym:
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
    return loss


