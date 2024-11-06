import argparse
from torch import optim
from MATE import *
from utils import *
import warnings
import random
from tqdm import tqdm
from torch_geometric.data import Data
from MATE import  NewGConv, NewNewEncoder, NewNewGRACE
from sklearn.cluster import KMeans
from xxx import *
from Visualization import t_SNE, Visualization


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')#cora  citeseer amac amap
parser.add_argument('--method_name', type=str, default='Model')
parser.add_argument('--topK_list', type=list, default=[10, 20, 50])
parser.add_argument('--seed', type=int, default=72)#72
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)#原来0.001c
parser.add_argument('--weight_decay', type=float, default=5e-5)  
parser.add_argument('--train_fts_ratio', type=float, default=0.4)
parser.add_argument('--generative_flag', type=bool, default=True)
parser.add_argument('--cuda', action='store_true',
                    default=torch.cuda.is_available())

parser.add_argument("--layer", nargs="?", default="gcn",
                    help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu",
                    help="Activation function for GNN encoder, (default: elu)")
parser.add_argument("--activation", nargs="?", default="relu",)
parser.add_argument('--encoder_channels', type=int, default=128,
                    help='Channels of GNN encoder layers. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=64,
                    help='Channels of hidden representation. (default: 64)')
parser.add_argument('--decoder_channels', type=int, default=32,
                    help='Channels of decoder layers. (default: 32)')
parser.add_argument('--encoder_layers', type=int, default=2,
                    help='Number of layers for encoder. (default: 2)')
parser.add_argument('--eproj_layer', type=int, default=2,
                    help='Number of layers for edge_projector. (default: 2)')
parser.add_argument('--decoder_layers', type=int, default=2,
                    help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8,
                    help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument('--eproj_dropout', type=float, default=0.2,
                    help='Dropout probability of edge_projector. (default: 0.2)')
parser.add_argument('--decoder_dropout', type=float, default=0.2,
                    help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument('--bn', type=bool, default=False)
parser.add_argument('--device', type=int, default=0, help='CUDA')
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--temp', type=float, default=0.2)


def main(args):
    set_random_seed(args.seed)#72
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)

    Adj, Diag, Ture_feature, A_temp = input_matrix(
        args, adj, norm_adj, true_features)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj) #1083，271，1354，1625且是打乱后的


    x_view1_ = true_features
    x_view1_[vali_test_id] = 0.0 #torch.Size([1625, 1433]),vali_test_id中这些点的属性赋为0
    data_1 = Data(x=x_view1_, y=node_labels, edge_index=indices)
    #data_1_1 = Data(x=x_view1_, y=node_labels, edge_index=indices_1)

    x_view_ = true_features
    x_view_[vali_test_id] = 0.0

    '''
    temp_adj = adj
    temp_adj = temp_adj.to_dense()
    for i in vali_test_id:
        tmpdata = torch.where(temp_adj[i, :] != 0)[0]
        tmpdata_2 = torch.where(temp_adj[tmpdata, :] != 0)[1]  # 用1没问题
        # tmpdata_3 = torch.where(adj[tmpdata_2, :] != 0)[1]
        a1 = []
        tmp_mean = torch.zeros(1, true_features.size()[1]).cuda()
        for q in range(len(tmpdata)):
            if tmpdata[q] in train_id:
                a1.append(tmpdata[q])
        if (len(a1) > 0):
            tmp_mean = torch.zeros(1, true_features.size()[1]).cuda()
            a4 = torch.tensor(x_view_[a1, :])
            for w in range(true_features.size()[1]):
                tmp_mean[:, w] = torch.mean(a4[:, w])

        a2 = []
        tmp_mean_2 = torch.zeros(1, true_features.size()[1]).cuda()
        for e in range(len(tmpdata_2)):
            if tmpdata_2[e] in train_id:
                a2.append(tmpdata_2[e])
        if (len(a2) > 0):
            tmp_mean_2 = torch.zeros(1, true_features.size()[1]).cuda()
            a4 = torch.tensor(x_view_[a2, :])
            for w in range(true_features.size()[1]):
                tmp_mean_2[:, w] = torch.mean(a4[:, w])

        x_view_[i, :] = 2 * tmp_mean + 4 * tmp_mean_2
        '''

    x_view_ = x_view_.cuda()
    #diff = diff.cuda()
    #x_view2 = x_view_.cuda()
    x_view2 = torch.mm(norm_adj.cuda(), x_view_).cpu() #扩散矩阵*把真是特征中验证测试集节点特征赋为0的特征矩阵  #这里应该可以改
    #x_view2 = torch.mm(diff, x_view_).cpu()
    data_2 = Data(x=x_view2, y=node_labels, edge_index=indices)
    #data_2 = Data(x=x_view_, y=node_labels, edge_index=indices)
    #data_2_2 = Data(x=x_view2, y=node_labels, edge_index=indices_1)
    fts_loss_func, pos_weight_tensor, neg_weight_tensor = loss_weight(
        args, true_features, train_id)

    set_random_seed(args.seed)
    mask = MaskEdge(p=args.p) #随机隐藏一些边
    #mask_1 = MaskEdge(p=0.9)
    #feature_learn = Feature_learner(true_features.size()[0], true_features.size()[1])
    feature_learn = Feature_learner(x_view1_,adj,
        true_features.size()[0], true_features.size()[1],vali_test_id,train_id)
    #feature_learn_1 = Feature_learner_1(diff,x_view1_, adj,
                                    #true_features.size()[0], true_features.size()[1], vali_test_id, train_id)
    #encoder = GNNEncoder(data_1.num_features, args.encoder_channels, args.hidden_channels, num_layers=args.encoder_layers, dropout=args.encoder_dropout,bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.eproj_layer, dropout=args.eproj_dropout)

    projector = Projector(args.hidden_channels, args.encoder_channels, out_channels=data_1.num_features,
                          num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    con_projector = Con_Projector(args.hidden_channels, args.encoder_channels, out_channels=data_1.num_features,
                          num_layers=args.decoder_layers, dropout=args.decoder_dropout)
    #ADJ=0
    #activation = 'prelu'
    #num_hidden = 256
    Newencoder = NewNewEncoder(data_1.num_features, args.num_hidden,args.activation,base_model = NewGConv, k = 2).cuda()  # 新的编码器
    MAmodel =  NewNewGRACE(Newencoder,  num_hidden=512, num_proj_hidden = 512, tau=0.4).cuda() #原本tau=0.4

    #ms = MAmodel(x1, x2, x3, x4)
    model = Model(edge_decoder, projector, con_projector,args.temp, pos_weight_tensor, neg_weight_tensor,MAmodel, mask)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    optimizer_learner = torch.optim.Adam(
        feature_learn.parameters(), lr=1e-3, weight_decay=args.weight_decay)


    def scheduler(epoch): return (
        1 + np.cos((epoch) * np.pi / args.epoch)) * 0.5 #结果在0到1之间
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=scheduler) #这行代码使用 PyTorch 中的学习率调度器 LambdaLR。LambdaLR 允许你通过一个自定义的 lambda 函数来调整学习率。lr_lambda: 这是一个用于动态调整学习率的 lambda 函数。scheduler 可以是一个函数或一个 lambda 表达式，该函数的输入是当前的 epoch 数，输出是相应的学习率


    if args.cuda:
        data_1 = data_1.cuda()
        data_2 = data_2.cuda()
        model = model.cuda()
        feature_learn = feature_learn.cuda()
        #feature_learn_1 = feature_learn.cuda()
        norm_adj = norm_adj.cuda()

    eva_values_list = []
    best = 0.0
    print('---------------------start trainning------------------------')
    for epoch in tqdm(range(1, 1 + args.epoch)): #这段代码使用了 tqdm 函数库中的 tqdm 函数，它用于在循环中创建进度条，以便在控制台中显示循环的进度。
        model.train()
        feature_learn.train()
        #feature_learn_1.train()
        loss = model.train_one_epoch(data_1, data_2,
                                     #data_1_1,data_2_2,
                                     norm_adj, feature_learn,
                                     train_id, vali_test_id,epoch,args.epoch,adj)
        
        optimizer.zero_grad()
        optimizer_learner.zero_grad()
        #optimizer_learner_1.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_learner.step()
        #optimizer_learner_1.step()
        scheduler.step()
        if epoch % 20 == 0:
            model.eval()
            feature_learn.eval()
            #feature_learn_1.eval()
            with torch.no_grad(): #with torch.no_grad(): 是 PyTorch 中的上下文管理器，用于在其范围内禁用梯度计算。当进入 with torch.no_grad(): 代码块时，PyTorch 不会跟踪在其范围内发生的操作，也不会计算任何操作的梯度。这对于在推断或验证阶段执行不需要梯度的计算非常有用，以提高效率并减少内存消耗。
                X_hat = model(data_1, data_2, norm_adj, 
                              feature_learn,train_id, vali_test_id)
            gene_fts = X_hat[vali_id].cpu().numpy()
            gt_fts = Ture_feature[vali_id].cpu().numpy()
            avg_recall, avg_ndcg = RECALL_NDCG(
                gene_fts, gt_fts, topN=args.topK_list[2])
            eva_values_list.append(avg_recall)
            if eva_values_list[-1] > best:
                torch.save(model.state_dict(), #model.state_dict() 返回一个字典，其中包含了模型的所有参数（权重和偏置）以及对应的键。具体而言，字典的键是参数的名称，而对应的值是包含参数值的张量。
                           os.path.join(os.getcwd(), 'model',
                                        'final_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio)))  #: os.getcwd()获取当前工作目录的路径。os.path.join(...): 使用 os.path.join 将路径组合成文件路径。torch.save(...): 将模型的状态字典保存到指定的文件路径中。
                torch.save(feature_learn.state_dict(), os.path.join(os.getcwd(), 'model',
                        'ft_learn_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio)))
                #torch.save(feature_learn_1.state_dict(), os.path.join(os.getcwd(), 'model',
                         #'ft_learn_1_model_{}_{}.pkl'.format(args.dataset,args.train_fts_ratio)))
                best = eva_values_list[-1]


    model.load_state_dict(
        torch.load(os.path.join(os.getcwd(), 'model', 'final_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))
    feature_learn.load_state_dict(
        torch.load(os.path.join(os.getcwd(), 'model', 'ft_learn_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))
    #feature_learn_1.load_state_dict(
        #torch.load(os.path.join(os.getcwd(), 'model', 'ft_learn_1_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))

    model.eval()
    feature_learn.eval()
    #feature_learn_1.eval()
    _,_ = test_model(args, model, norm_adj, feature_learn, Ture_feature,
                                data_1, data_2, train_id, vali_id, vali_test_id, test_id)
    with torch.no_grad():
        x_hat = model(data_1, data_2, norm_adj, feature_learn, train_id, vali_test_id) #经过了GCN得到的
        gene_data = x_hat[test_id]
        labels_of_gene = node_labels[test_id]

        class_num = max(node_labels.numpy()) + 1
        kmeans = KMeans(n_clusters=class_num, n_init=20).fit(x_hat.cpu())
        acc, nmi, ari, f1 = eva(node_labels.numpy(), kmeans.labels_, 0)
        ''' 

        output = x_hat.cpu().detach().numpy()
        labels = node_labels.cpu().detach().numpy()
        result = t_SNE(output, 2)
        Visualization(result, labels)
        '''
    #adj = adj.to_dense()
    #test_X(gene_data.cpu().numpy(), labels_of_gene.cpu().numpy())
    #test_AX(gene_data.cpu().numpy(), labels_of_gene.cpu().numpy(), adj[test_id, :][:, test_id].cpu().numpy()) #注意看看怎么采样子边生成测试图的
    
if __name__ == "__main__":
    args = parser.parse_args() #parser 是一个 ArgumentParser 对象，通过它可以定义命令行参数的类型、名称、帮助信息等。parse_args() 方法用于解析命令行参数，并将其存储在一个 Namespace 对象中，该对象的属性名对应参数的名称，属性值对应参数的值。
    args = load_best_configs(args, "configs.yml")
    print(args)
    torch.cuda.set_device(f'cuda:{args.device}')
    main(args)


