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


#cora n_init=1:  Epoch_0 :acc 0.7230 , nmi 0.5475 , ari 0.5088 , f1 0.7006. acc 0.7201 , nmi 0.5414 , ari 0.4952 , f1 0.6994; Epoch_0 :acc 0.6222 , nmi 0.5015 , ari 0.3803 , f1 0.6476；Epoch_0 :acc 0.6233 , nmi 0.5031 , ari 0.3838 , f1 0.6461；Epoch_0 :acc 0.6307 , nmi 0.4926 , ari 0.3825 , f1 0.6478； n_init=20:  acc 0.6407 , nmi 0.5100 , ari 0.4000 , f1 0.6552 ；acc 0.6333 , nmi 0.5074 , ari 0.4151 , f1 0.6311
#citeseer n_init=20:  Epoch_0 :aacc 0.6090 , nmi 0.3499 , ari 0.2926 , f1 0.5702 acc 0.5482 , nmi 0.3070 , ari 0.2350 , f1 0.5337
#amac n_init=20:acc 0.3861 , nmi 0.2567 , ari 0.1762 , f1 0.22648;acc 0.3378 , nmi 0.1865 , ari 0.1261 , f1 0.1989
#amap n_init=1:Epoch_0 :acc 0.3556 , nmi 0.2133 , ari 0.0759 , f1 0.2510; n_init=20:Epoch_0 :acc 0.4082 , nmi 0.2760 , ari 0.0863 , f1 0.3029; acc 0.4016 , nmi 0.2804 , ari 0.0842 , f1 0.2964
#13 26 27
#diff换为norm_adj cora:0.17539 0.2373 0.2492 0.28719 0.3828 0.35779  0.8476 0.86389; 0.1741 0.2376 0.25016 0.28795 0.38126 0.3577 0.8453 0.8629；0.17529 0.23903 0.25019 0.28911 0.38218 0.35915 0.84468 0.86286;0.1751 0.23838 0.24906 0.28788 0.38040 0.35766 0.84542  0.86448;0.17471 0.23769 0.2490 0.2872 0.3801 0.35686 0.84283
    #去掉特征0.17254 0.2362 0.24585 0.28571 0.38089 0.35723 0.84662 0.86382 #消融去掉我的创新点：0.17123 0.23225 0.24355 0.28056 0.37454 0.35103 0.8354 0.85443；消融长短编码器：0.17253 0.23673 0.25179 0.28907 0.38175 0.35786 0.84302 0.85389
#diff换为norm_adj citeseer:0.1011 0.17449 0.16172 0.22338 0.27749 0.29961 0.67589 0.69053; 0.68773  0.69599;0.6762 0.69138;0.09986 0.17125  0.1608 0.2222 0.2737 0.2964 0.69001 0.6946;0.68899 0.6956
    #0.10056  0.17152 0.15994 0.22336 0.2756 0.2983 0.67649 0.6903 #消融去掉我的创新点：0.09843 0.16855 0.15568 0.21643 0.268219 0.29037 0.67589 0.68958;消融长短编码器：0.09799 0.1688 0.15526 0.2167 0.2694 0.29176 0.6786 0.6909
#diff换为norm_adj amac:0.0462 0.1126 0.08058 0.15806 0.16397 0.24706  0.889048 0.8997  ;0.04601 0.1123 0.08034 0.1577 0.1636 0.2454 0.88536 0.89712; 0.04595 0.11213 0.08032 0.15758 0.16270 0.24565 0.888612 0.89876; 0.04537 0.1104 0.07925 0.15535 0.16329  0.24478 0.8832 0.89472;  0.04637 0.11287 0.08050 0.15811 0.16386 0.24708 0.8828 0.89661
    #消融去掉特征0.0451 0.11002 0.07902 0.15495 0.16182 0.24334 0.88126 0.89509# 消融去掉我的创新点：0.045326 0.11034 0.07934 0.15538 0.162205 0.24382 0.88337 0.89470; 0.043927  0.10746 0.077043  0.15143 0.15958 0.23940 0.87276 0.892437;#消融长短编码器：0.04491 0.10967 0.07858 0.15432 0.16143 0.24272 0.87422 0.89133
#diff换为norm_adj amap:0.0450 0.1096 0.08015 0.1562 0.1660 0.248 0.92165 0.926607;0.04468 0.10913  0.07971 0.15564 0.16588 0.24748 0.92141  0.92515; 0.04475 0.1094 0.0798 0.15598 0.16548 0.24738 0.92039 0.92541
    #0.04445 0.10886 0.07973 0.15561 0.16539 0.24698 0.91869 0.92366 #消融去掉我的创新点：0.04402  0.10769 0.07925 0.15439 0.165478 0.246282 0.91545 0.92522;0.04477 0.10937 0.07989 0.15603 0.16537 0.24724 0.91756 0.92611 消融长短编码器：0.0448 0.1094 0.08012 0.1563 0.16583 0.2478 0.91717 0.9260;0.0447,0.1092 0.0799 0.1561 0.1657 0.2473

#cora换成512，256：0.17977 0.24782 0.251746 0.296246 0.3813 0.3646 0.85465 0.8624 acc 0.6728 , nmi 0.5316 , ari 0.4426 , f1 0.6736

#cora:mlp=64,hi=3,把tmp_mean初始化去掉,应该是1+0.2,final=false;tau=0.4  #2,2.10,10看上去效果更好： 0.37921  0.35644  0.84549  0.8669； 0.37839 0.35642 0.84453 0.86847
#新0.37606 0.35452 0.84793 0.862563 #新新，理论更合理：2mean+4mean2,2.0+1.0: 0.3776 0.3557 0.84586  0.8571;0.17137 0.23666 0.24977 0.28814 0.37825 0.35625 0.84748 0.85828
        #信信信，hi=1,只要featureloss,0.4缺失:0.17308 0.23701 0.249138 0.288113  0.379003 0.356213 0.850074  0.85936 ;0.1705  0.2352  0.24875 0.28762 0.37861 0.3558 0.84926 0.86353; 0.172548 0.23681 0.25068  0.28889 0.37545 0.35520 0.849188 0.856734想尝试换掉原编码器跑一下（不行）：0.1716 0.2365  0.24645 0.2863 0.37693 0.35537 0.84689  0.85149 ss
                #消融用原编码器：0.17087 0.23602 0.24624  0.28635 0.37626 0.35526 0.84681  0.85223;消融去掉特征损失函数：0.17316 0.23714 0.24976 0.28846 0.37563  0.35528 0.84896 0.85799;0.166108 0.22741 0.24021 0.27729 0.37113 0.34656 0.84135 0.85681;消融0参：0.13076  0.17989 0.1870 0.21756 0.30034 0.2775 0.64128 0.79152
            #缺失0.1：0.12795 0.17946 0.19175  0.22184 0.30546 0.28197 0.83276 0.858048 ；缺失0.2：0.15926  0.22363 0.22439 0.266847  0.34618  0.3310 0.84451  0.86207
                #缺失0.05：0.11882 0.17202 0.18121  0.213407 0.286827 0.26988  0.80352 0.854205;0.12157 0.17668 0.18076 0.21616 0.28735 0.27305 0.803306 0.85429
    #0.05,0.1,0.5缺失率下：原编码器0.29584 0.28193 0.81587 0.86012是最好的;0.05,0.1,0.85原编码器缺失率:0.1242 0.18096 0.17608 0.21606 0.28611 0.27437 0.7920 0.85542；0.05,0.1,0.85改编码器缺失率: 0.12794 0.18195  0.17654 0.214917  0.27921 0.26929 0.77376 0.84808
#amac0.5+0.2,tau=0.3,hi=1
#citeseer:0.09899  0.16940 0.1586 0.2193 0.2746  0.29557 0.68461 0.6927
    #消融用原编码器:0.0991  0.16905 0.15608 0.21672 0.2679 0.29017 0.68202 0.68772;消融去掉特征损失函数(x和A+X跑的有点好):0.09805 0.16806 0.15697 0.2174 0.2720 0.29309 0.68611 0.69529; 0.07278 0.12283 0.12179 0.1636 0.2244 0.2307 0.6042 0.6618
#amap:0.04463 0.109136 0.07953 0.15547  0.165145 0.24678 0.92088  0.92535;0.04492  0.1096 0.07995 0.15615 0.16566 0.2476 0.92002 0.92632
    #消融用原编码器：0.04481 0.10966 0.07993 0.15632 0.16590 0.24798 0.91560 0.9255;消融去掉特征损失函数：0.04481 0.1094 0.07962 0.1557 0.1649  0.2467 0.92196  0.92559。0.04488 0.10962 0.07981 0.15607 0.16522 0.24730  0.91845  0.92603;消融0参：0.03606  0.0896 0.0659 0.1299 0.1444 0.2140 0.76 0.9071

#begining test_X......classification performance: 0.8418745387453874  0.8394372010386771  0.8420964876315431  0.8397324039907066   0.8406920869208692
#begining test_AX......GCN(A+X),  avg accuracy: 0.8579882465491322  0.8527464807981413  0.8588000546672134  0.8522285089517562   0.8586508131747985
#avg_recall:  0.1714645329583666  0.24407036253229825  0.37349179772196367
#avg_ndcg:  0.2362642339966447  0.28491966459256435  0.3530860461969298
#重新跑的topK: 10, recall: 0.1712, ndcg: 0.23620    topK: 20, recall: 0.24289 ndcg: 0.28402  topK: 50, recall: 0.37423, ndcg: 0.3533      0.84135  0.85702
#topK: 50, recall: 0.37379182019163026, ndcg: 0.35309205452161624    0.8404720513871805  0.8570225502255022
#0.1randn的原（x会很好，a+x会很差：topK: 50, recall: 0.3750413666662332, ndcg: 0.35528753962089543   0.8471903785704523   0.853781057810578  #topK: 50, recall: 0.37660012814077765, ndcg: 0.35610045980760224   0.846376930435971   0.8501626349596829 # 0.846745934126008   0.8504592045920459


#加了扩散矩阵的索引（跑了半小时）
#topK: 10, recall: 0.1698447169804739, ndcg: 0.23332040135080934     #topK: 20, recall: 0.24315993980309297, ndcg: 0.2825504402228344      #topK: 50, recall: 0.37597317931762164, ndcg: 0.3522612074890631
#begining test_X......classification performance: 0.8434246275796091
#begining test_AX......GCN(A+X),  avg accuracy: 0.8533374333743338

#长短编码器
#topK: 10, recall: 0.16874851339122734, ndcg: 0.23309587152679437       #topK: 20, recall: 0.2431248692920831, ndcg: 0.2822412817810818        #topK: 50, recall: 0.37347078840813697, ndcg: 0.35130766543322756

#topK: 10, recall: 0.1679200335874786, ndcg: 0.2317777572471218          #topK: 20, recall: 0.24264210304878855, ndcg: 0.28145289870162765      #topK: 50, recall: 0.37481209110122427, ndcg: 0.35114925430098465
#classification performance: 0.8369246959136257   GCN(A+X),  avg accuracy: 0.8562164821648217

#自己完全改了MA框架的：
# [2,2],[8,8]begining test_X......classification performance: 0.3852986196528632   #begining test_AX......GCN(A+X),  avg accuracy: 0.7564414377477108
#[2,2],[2,2]   #0.32282520158534916    #0.705990979909799

    #加了drop out和激励
    #topK: 10, recall: 0.1553446815009843, ndcg: 0.21632146534157104   #topK: 20, recall: 0.22371437037918407, ndcg: 0.26263740061697965  #topK: 50,topK: 50, recall: 0.3589899579260435, ndcg: 0.33299295022803416

    #begining test_X...... classification performance: 0.795348913489135    #begining test_AX......   GCN(A+X),  avg accuracy: 0.8421115211152111
    #0.7930583572502392    0.843593549268826    #0.7963785704523711   0.8510504305043052

    #卷积层数变了一下
    #topK: 10, recall: 0.15489058666019145, ndcg: 0.21545269170120948    #topK: 20, recall: 0.22208466175123998, ndcg: 0.2608625909109646   #topK: 50, recall: 0.3513839023904386, ndcg: 0.32941015974344445
    #classification performance: 0.8042091020910209  GCN(A+X),  avg accuracy: 0.8369492961596283    #0.8107806478064781   0.8402667760010933

    #又去掉dropout和activate，就是只加了对比
    #topK: 10, recall: 0.17082923208194062, ndcg: 0.23426723090930127  #topK: 20, recall: 0.24576739255276128, ndcg: 0.28380061342403395  #topK: 50, recall: 0.3759794818571929, ndcg: 0.3528461189697144
    #topK: 10, recall: 0.1701881533803564, ndcg: 0.23394928651521124      topK: 20, recall: 0.24618996058299752, ndcg: 0.2839410936303539      topK: 50, recall: 0.37622082315853517, ndcg: 0.3529507434675585    #topK: 50, recall: 0.37544100910715694, ndcg: 0.3526219450725622
    #classification performance: 0.8364059040590407  GCN(A+X),  avg accuracy: 0.8543649036490365
   #0.8378100314336476   0.8559915265819324   #0.8367762744294109   0.8542905562388958
    #hi变成了4（跑的真不错）  topK: 10, recall: 0.1718 ndcg: 0.2353  topK: 20, recall: 0.2458, ndcg: 0.2844  topK: 50, recall: 0.3754 ndcg: 0.3532   0.83921   0.86183   #topK: 50, recall: 0.3766ndcg: 0.354408    0.83788   0.8585097   #topK: 50, recall: 0.3752215445869954, ndcg: 0.3533302233040233   0.8378860188601885    0.8622011753450867  #0.8370739374060407  0.8607984146508132

    #改了参数初始化和对比
    #topK: 10, recall: 0.1714864055158995, ndcg: 0.23557860264481115   #topK: 20, recall: 0.2457975714928489, ndcg: 0.2858104897434251  #topK: 50, recall: topK: 50, recall: 0.3782893033690591, ndcg: 0.3558157142432529     topK: 50, recall: 0.37836166281717154, ndcg: 0.3558277772762786
    #classification performance: 0.8420992209922099    GCN(A+X),  avg accuracy: 0.8567276206095394   #0.8428375017083504    0.857762470958043
# hi为4   0.17272415691377183, ndcg: 0.2370646564653415   0.24761764310857098, ndcg: 0.28688571547801917    topK: 50, recall: 0.3767741580107402, ndcg: 0.35546486636640245    classification performance: 0.8361894218942189  GCN(A+X),  avg accuracy: 0.8577684843515101    #topK: 50, recall: 0.3764079219922016, ndcg: 0.3555184279969   0.8342692360256937    0.854223315566489  #topK: 50, recall: 0.3760380335254742, ndcg: 0.3552943589975375    0.8329408227415607   0.8545950526171928
#hi为4，参数是改了加原有：topK: 10, recall: 0.1727491524625001, ndcg: 0.2374692724486982   recall: 0.2477247074594303, ndcg: 0.2870166608875501  topK: 50, recall: 0.3790069218751497, ndcg: 0.3565639488590117     classification performance: 0.8377373240399072    0.8605023916905836   #topK: 50, recall: 0.3787591776880746, ndcg: 0.35660894315798203   0.8373683203498702
   #0.1的randn(做了没有改参的0.1randn，0.8333811671450049  0.8606617466174662  #0.8398819188191883，0.8623550635506355  #0.8405472188055214，0.8613196665299986）: topK: 50, recall: 0.3785132494687714, ndcg: 0.3562440265776891   0.8423944239442394   0.8599844198441986  #topK: 50, recall: 0.37743, ndcg: 0.3558401   0.840988  0.861826 #topK: 50, recall: 0.37770, ndcg: 0.35616   0.84327 0.8626412
    #同上：topK: 10, recall: 0.1725 ndcg: 0.237 topK: 20, recall: 0.2469 ndcg: 0.2873  topK: 50, recall: 0.377, ndcg: 0.356   0.844904  0.862642   #0.8392  0.85924  #0.8459  0.85968  #0.84453   0.86330  #0.84505  0.86315 #0.84039  0.86249 # 0.83995   0.86278   0.85939
    #之前的参数初始化都错了：topK: 10, recall: 0.17201  ndcg: 0.23703    recall: 0.247555   ndcg: 0.28753   recall: 0.37600


#另一种改参 （跑出的错误改参：topK: 50, recall: 0.37791    ndcg: 0.3562  0.8446095   0.86338）ttopK: 50, recall: 0.3778877 ndcg: 0.356103   0.838477   0.85614  #(1randn)ll: 0.377991   dcg: 0.355   0.84039   0.8593  #都是1：topK: 50, recall: 0.378417   ndcg: 0.3556   0.841210   0.86168
    #一阶+二阶一起：topK: 50, recall: 0.37757   ndcg: 0.35595   0.84372   0.85858  #0.37790    0.35679  0.8420   0.8531  #recall: 0.37697   0.35610   0.842761   0.856293  #(（2一阶，1二阶： 0.37776  ndcg: 0.3553   0.84608   0.85511   #0.378420   0.35554   0.84719   0.8553  #改为0.5randn：0.3789   0.35716  0.84482   #(#0.2randn:0.37853  0.356358   0.84859   0.85658    #0.37796   ndcg: 0.35633   0.84962   0.856215
        #三阶：（2，1，1：topK: 10, recall: 0.17590   0.23854   ll: 0.24954   0.28781   ll:0.38073   0.35721   0.84652    0.85356）（2.1.0.5+0.1randn:0.3788   0.35665  0.84712   0.8582  #2.1.0.5,0.2  ll:0.37947   0.3577   0.8474    0.85540  (#2,1,1,0.2    0.37915  .35754  0.84933   0.85407   #0.38037   0.35717  0.84815   0.85621 )#2,1,1,0:0.37797   0.3561  0.84822   0.85363  #2,1,1,0.05:0.37768   0.3566   0.84793   0.85466
            #(2,2,1,0.2:0.37926   0.35714  0.8460   0.8596  ;0.37990  0.3568  0.8463   0.85607)  (3,2,1,0.2:0.37992  0.356609  0.8468  0.85371)  (1,1,1,0.2: 0.38037   0.3584   0.84490   0.8611 ;  0.37831  0.3571   0.8449   0.8522 ; 0.3787  0.3574  0.8447   0.8579 ; .0.3800  0.3579  0.8463  0.8619;  0.379377   0.35765   0.84556   0.8579
        # 第二张图的属性由归一化邻接矩阵乘缺失属性变为随机属性：0.37829    0.3569   0.850147   0.86124 ; 0.3776  0.3566  0.84748   0.8569
        #第二张图的属性由归一化邻接矩阵乘缺失属性变为一阶属性：0.37734  0.3581  0.8424  0.8640；0.3779   0.3584  0.8417  0.8637；0.37779   0.3584  0.8454   0.8654
#隐藏层128->256:0.3777  0.36152  0.85199  0.85725



#改了参数+原有对比做消融（反应了比原有参数初始化好，但是我的对比算法好像不太行）
    #topK: 10, recall: 0.17346983997663284, ndcg: 0.23830527593554296   #topK: 20, recall: 0.2490931348400923, ndcg: 0.2882235453525295    #topK: 50, recall: 0.37741539975831, ndcg: 0.3562957098636924
    #classification performance: 0.8435760557605576  GCN(A+X),  avg accuracy: 0.8528954489544895   #0.8446852535192019   0.8540002733360668
    #结合参数：topK: 50, recall: 0.37824, ndcg: 0.355347251  0.84202  0.85068 #topK: 50, recall: 0.3791, ndcg: 0.3560  0.84202   0.85429  topK: 50, recall: 0.37830 ndcg: 0.3558  0.8420    0.85518


#citeseer:0.2570  0.2775  0.6719   0.6857; 0.2560  0.2749  0.6643   0.6793 #125->256 ：0.27152  0.29608  0.68076  0.6866;0.2702  0.2945   0.67848  0.6845;#换成prelu:0.27096   0.2955  0.68058   0.6853 ;0.2699  0.2943  0.6829  0.6941;0.2705  0.2956  0.6811  0.6877
   #原来的编码器：0.25804  0.2772   0.68238   0.69810
    #不改第二个图的参数了，class_eva的隐藏层改为128：0.1018  0.1745    0.1605  0.2237   0.2776   0.3006   0.6831   0.6933；256：0.2763   0.2992   0.6832   0.6950
    #0.27771  0.30176  0.68232  0.6929



#(之前的不准）amac:0.044285  0.1077    0.07757  0.1520    0.16075  0.2405   0.8636   0.8876  (新的跑出来不好：0.0426  0.10388  .07513  0.14715  0.1557  0.2334  0.81704  0.8718）(改参：0.0450  0.1100   0.0792  0.1553  0.16254  0.2442  0.86874  0.8904）（256，128的层：0.0447  0.1090  0.0783  0.15360  0.16152  0.2423   0.8654  0.8876）（cleass_eva变128：0.1609  0.2410  0.8578   0.8865）（h改为2：0.1605  0.2411  0.8559  0.8851）
#接上:((换位原来初始化参：0.1627  0.24508  0.87722  0.89354;0.1619 0.2435 0.8697）（256，128，125,h=6:  0.1603   0.2413  0.87642   0.8941))  (randn换为1,很差:0.15764  0.2359  0.8305  0.87770) (把mean初始化提前到for前面：0.1563  0.2363  0.8471  0.8833）（没有randn，即randn为0：0.16259  0.24439  0.86609  0.8898）（0.5self.x+0.5randn:0.04554  0.11065  0.07941 0.15554  0.16215  0.24396  0.86841  0.89083)（0.5self.x+0.1randn:0.04567  0.11103  0.0797  0.15612  0.16245  0.24453  0.87436  0.89207（在之中化为lr=0.0005，效果一样0.16239  0.24446  0.87318  0.89253; 没要第二个mean:0.16272  0.24441  0.87034  0.89028，lr=0.001s时没第二个mean: 0.16197  0.2431 0.86892 0.8880）（0.25+0.05：0.0452  0.1103  0.16267  0.0791  0.15529  0.2442  0.87332  0.89140）（lr为0.002很差，不到八十）
#接上：（0.5+0.2：0.16288  0.24408  0.87278  0.89031)(将hi=2：1+0.1:r+n效果正常，0.867511  0.89195，0.5+0.1：跑的不好，1+0.2：都很不好，0.5+0.2：0.16195  0.24425  0.8744  0.89029）（参数和对比都是原来：0.04516 0.11027  0.07929 0.15545 0.16198  0.243803 0.87757 0.89361）(只有对比是原来：0.16157  0.24294  0.87079  0.89002）(超参384，128，hi=3：0.04528 0.1102 0.07967 0.1556 0.16268  0.24431 0.87652 0.8909)(
#(由cora重改的amac:hi=1,0.5+0.2,hid=64：0.16180 0.24229 0.87356 0.88880;  0.16184 0.24208  0.87406  0.88901;0.1618  0.2421 0.8750 0.8885; （hi=2,0.5+0.2:0.161678 0.24326 0.8750 0.89021;:0.16247 0.24440  0.87408 0.89077(512+256+256,hi=1:0.16236 0.24401 0.87120 0.88999;0.16187 0.24319 0.86790)
#hi=1,1.0+0.2,hid=64,decoderchannel=16,tau=0.3,activation=prelu:0.04539,0.11063 0.07929 0.15557 0.16229 0.24421 0.874127 0.89179；0.5+0.2：0.16105 0.24332 0.8766  0.89198;和amac参数一样不太行；新configure0.16218 0.24424 0.87722 0.89205;改个0.5+0.2：0.045495  0.11081 0.07966 0.1560495 0.16256 0.244565 0.87769 0.892175
    #新新配置，hi=1,featureloss太好了:0.04569 0.11136  0.079665 0.15636 0.163125 0.24529 0.885021 0.896975;0.04543 0.11061 0.07940  0.15559 0.161658  0.24351 0.88388 0.89649  ;0.045536   0.11099 0.07945 0.155936 0.16269 0.24475 0.883609 0.89627
        #消融去掉特征对比：0.04405 0.10722  0.07734 0.15139  0.16038 0.23980 0.879305 0.88976；消融去掉编码区： 0.04488 0.10960 0.07838 0.15405 0.159946   0.24137 0.86287 0.88912; #0.03080 0.07478 0.05847 0.11208 0.12855 0.18831  0.55151 0.79936

 #0.05效果x,a+x不好:0.03995 0.09824 0.07101 0.13967  0.14950 0.22357 0.845568 0.89151;     0.05,0.1,0.5的切换：0.0414716  0.1018  0.072596  0.143429 0.150639  0.226957  0.861911  0.887346 #换了原来的编码器，明显提高，下一步准备再试试和原超参一起的影响：0.15074 0.22617 0.85796  0.89457;加了原超参，不行
    #新0.05，0.1，0.85：在featureloss基础上:0.03893 0.096219 0.069577 0.137217 0.147315 0.22040 0.846424  0.88575
#amap:0.0447  0.1094   0.0803  0.15648  0.16647  0.2482  0.91620  0.92368
#cora0.3792  0.3560  0.8457   0.8662;0.3791 0.35648 0.84719 0.86691 ;0.37833  0.35626 0.84113 0.86707;0.3776  0.35585  0.8461  0.8658;0.37788  0.3561 0.84512 0.86455 ； 0.37863  0.35508  0.84712  0.8661#for前加了全为0的mean0.3765    0.35509   0.8448  0.86448 #全为1:.3778  0.3555  0.84305  0.8665
#citeseer:0.10045  0.17172  0.15881  0.22065  0.27521  0.29709  0.68827  0.69715
##amac0.4缺失:0.04525  0.1103  0.07935  0.1555  0.16356  0.2451  0.87297  0.89224

#cora0.8:0.1765  0.2479 0.2636 0.3049 0.4055 0.3819 0.8612 0.8584
    #0.7:0.1793 0.2462 0.2587 0.2996 0.3995 0.3744 0.8533 0.8497
    #0.6:0.1763  0.2421 0.2534 0.2933 0.3891 0.3652 0.8463  0.8441
    #0.5:0.1716  0.2359 0.2509 0.2883 0.3845 0.3592 0.8425 0.8440
    #0.4:0.1748 0.2372 0.2493 0.2872 0.38187 0.3572 0.84608 0.8668
    #0.3:0.1693 0.2328 0.2431 0.2820 0.3718 0.3498 0.84929 0.8668
    #0.2: 0.16123 0.2252 0.2277 0.2695 0.3489 0.3333 0.84709 0.86407
    #cora0.1缺失:0.1360  0.1974  0.1976   0.2381  0.30725  0.2960  0.8475  0.8665
    #0.05:0.1279  0.1794  0.1779  0.2135  0.2824  0.2691  0.79213  0.86623 #mean初始化为一个train的属性0.1273  0.1794  0.1787  0.2143  0.2824  0.2695  0.7901  0.86528）
    #0.01:0.0984  0.14458 0.1426 0.17429 0.2268 0.21847 0.6249 0.80729

# citeseer0.8:0.1095 0.1856 0.1780 0.2431 0.3031 0.3257 0.6814 0.6808
    #0.7:0.11354 0.19139 0.17988 0.24716 0.30487 0.3296 0.6946 0.68557
    #0.6:0.1088 0.18468 0.17375 0.2389 0.2954 0.319046 0.69349 0.69714
    #0.5：0.1066 0.1805 0.1690 0.2327  0.2871 0.3105 0.6821 0.6850
    #0.3：0.0938  0.1603 0.1506 0.2078 0.2616 0.2807 0.6647  0.68477
    # 0.2: 0.07957 0.1351 0.1302 0.1775 0.2370 0.2477 0.6573 0.6798
    #0.1：0.0626  0.1046  0.109 0.14313 0.2102 0.2094 0.62103  0.6504
    #citeseer0.05:0.0570  0.0950  0.09974  0.13066  0.1850  0.1868  0.62719  0.7031
    #0.01:0.0398 0.06518 0.0728 0.0926 0.14829 0.1419 0.4430 0.5762

#amac0.2:0.0453 0.1103 0.07866 0.15465 0.15999 0.24141 0.8901 0.9054
    #0.1:0.04289 0.1050  0.0748 0.14759 0.15414 0.23239 0.881748 0.9028
    #0.05:0.0397 0.0978 0.0701 0.1385 0.1462 0.2203 0.8439 0.8848
    #0.01:0.0321 0.0793 0.0587 0.1155 0.1305 0.1926 0.8159 0.8654

#amap0.2:0.0447 0.1085 0.0783 0.15328 0.1619 0.24216 0.9204 0.92999
    #0.1:0.04156  0.1022 0.07387 0.1452 0.15316 0.2298  0.9046 0.92426
    #0.05:0.03799 0.0938 0.0653 0.13097 0.1264 0.1980  0.9033 0.92615
    #0.01:0.03046 0.07499 0.05567 0.10935 0.11695 0.17607 0.871846 0.91528

#cora 无学习有参2+0.2：0.174105 0.23717 0.25099 0.28842 0.38197 0.35765 0.84689 0.85496 ;0.1734 0.23638 0.2521  0.2884 0.3792 0.3560 0.84756 0.85725; 0.17445 0.23739 0.2511 0.28837 0.37978 0.35679  0.8500  0.85555
    #消融零参：0.17098 0.2364 0.2770 0.28695 0.37653 0.35514 0.84505; 0.17188 0.23406 0.2455 0.28347 0.37796 0.35345 0.84039  0.85104
        #消融特征：0.17399  0.23712 0.25185 0.28859  0.38087 0.35717 0.84719 0.85732；
#citeseer 无学习有参2+0.2:0.09557 0.16354 0.15318 0.21165 0.26809 0.28722 0.68257 0.69397
    #0.09568 0.16384 0.15375 0.21230 0.26989 0.28870 0.68142 0.68142


#超参分析：(1,2,4,6,8)(1,2,4,6,8)
    #cora1.1:0.1740 0.2356 0.2478  0.28535 0.38085 0.3554 0.8389 0.8621
    #1.4:0.17616  0.2381 0.25017 0.28763 0.3807  0.3566 0.84586 0.8650
    #1.8:0.1727 0.2356  0.2481 0.28596 0.38049 0.35641 0.84697 0.86160
    #1.16:0.1733 0.2359 0.2491 0.2865  0.37845 0.3549 0.8432 0.8552
    #2.1:0.1746 0.2370 0.2473 0.2861 0.3803 0.3561 0.8415 0.8621
    #2.4:0.1772 0.2392 0.2506 0.2885 0.3817 0.3579 0.84579 0.86219
    #2.8:0.1742 0.2373 0.2487 0.2871 0.3799 0.3567 0.8456 0.8619
    #4.1:0.1753 0.2370 0.2502 0.2871 0.3801 0.3562 0.8429 0.8637
    #4.4:0.17616 0.23798 0.25172 0.2886  0.38016 0.35669 0.84409 0.86138;0.17603 0.2379 0.25065 0.2880  0.3807 0.35689 0.8448 0.8621; 0.1759  0.2376  0.2507  0.2876  0.3808 0.3566 0.8480 0.8644
    #4.6:0.17527 0.23749 0.24988 0.28749 0.38073 0.35686 0.84837 0.8627
    #4.8: 0.1739 0.2360 0.24997 0.28737 0.3825  0.35725 0.8454 0.86315;  0.1756 0.23767 0.25026 0.2879 0.3826 0.3578 0.8466 0.86389
    #6.1:0.1749  0.2369 0.2489 0.2863 0.3805 0.3561 0.8445 0.8647
    #8.4: 0.1757 0.2375 0.2503 0.28748 0.3794 0.3560 0.8407 0.8593
    #8.8:0.17435  0.23738 0.25169 0.28868  0.38135 0.3575 0.8482 0.8638;0.17495  0.2371 0.25109 0.2880 0.3825 0.3576 0.8487 0.8630
    #8.12:0.1740 0.2369 0.2511  0.2882 0.3814 0.3574 0.8476 0.8618
    #8.16:0.17396 0.2363 0.25097 0.2876 0.38115 0.3568 0.8494  0.8597
    #12.16:0.1734 0.2360  0.2505 0.2878 0.38104 0.3568  0.8483 0.8598
    #:4,12:0.17542 0.2369 0.2495 0.2868 0.3809  0.3563 0.8488 0.8612
    #4.16:0.17377 0.23519 0.24847 0.28556 0.38073 0.35525 0.8459


#citeseer
    #0.0:0.1030  0.1766  0.1624 0.2262 0.2773 0.3018 0.6837  0.6954
     #1.1:0.1027 0.1763 0.16293 0.2266 0.2798 0.3034 0.6867 0.6994
    #1.4:0.1024 0.1754 0.1616 0.2250 0.2773 0.3010             0.6926
    #1.8：0.0984  0.1687 0.1579 0.2185 0.2720 0.2936 0.6879  0.6956
    #2.1：0.1029 0.1764 0.1643 0.2277 0.2811 0.3044 0.6867 0.6975
    #2.4:0.1019 0.1744 0.1624 0.2251 0.2794 0.3021  0.6903 0.7010
    #2.8:0.0996  0.1699 0.1584 0.2190 0.2735  0.2947 0.6872 0.6939
    #4.1:0.1018  0.1728 0.1645 0.2251 0.2792 0.3005 0.6900 0.6985
    #4.4:0.10076 0.1720 0.1634 0.2245 0.27839 0.30006 0.6880 0.6978;  编码器x_1不变：0.10279 0.1751 0.16365 0.22604 0.28039 0.30288 0.68587  0.6975
    #4.6:0.1002 0.17149 0.1612 0.2225  0.27658 0.298396 0.68659 0.6966
    #4.8: 0.0985 0.16913 0.1576  0.21867 0.27405 0.2951 0.6885  0.69721
    #6,1:0.1016 0.1729 0.1639 0.2250 0.2811 0.3020 0.6907 0.7006
    #6.4:0.09919  0.1695 0.1610 0.2212 0.2760 0.2968 0.6832 0.6936
    #6.8:0.0986 0.16899 0.1587 0.2193 0.2760 0.2963 0.6868 0.6964
    #8.1:0.1009 0.1716 0.1622 0.2229 0.2802 0.3006 0.6903 0.6988
    #8.4:0.1002 0.1704 0.1610 0.2212 0.2786 0.2986 0.6802  0.6945
    #8,8: 0.09808 0.16725  0.15909 0.2182 0.2752  0.29468 0.6838 0.6946

#amap
    #1.1:0.0441 0.1080 0.0792 0.1545 0.16545 0.2464 0.9157 0.92457
    #1.4:0.0442 0.1081 0.0792 0.1545 0.16519  0.2462 0.9172 0.92355
    #1.8:0.04445 0.10848 0.0792 0.1546 0.1645 0.24569  0.91558  0.924
    #4.4: 0.0450 0.1098 0.07993  0.1562 0.1655 0.2476 0.9189 0.9258； 编码器x_1不变：0.04449 0.1086 0.0795 0.1552 0.1656 0.2470 0.91958 0.9246
    #4.6:0.0449 0.10959 0.07975 0.15596 0.16569 0.2476 0.91945 0.92598
    #4.8:0.04505  0.10978  0.0796 0.1558  0.1650 0.2470 0.9200 0.9266
    #6:1:0.0445 0.1087 0.0795 0.1553 0.1655  0.2468 0.9179 0.9247
    #8.8:

#amac
    #1.1:0.0456 0.1113 0.0797  0.1565 0.1632  0.2455 0.8786 0.8931
    #1.4:0.0454 0.1109 0.0794 0.1560 0.1626 0.2447 0.8856 0.8974
    #1.8:0.0459  0.1120 0.0803 0.1576 0.1635 0.2462 0.8833 0.8964
    #4.1:0.0459  0.1121 0.0801 0.1574 0.1640 0.2468 0.8867 0.8964
    #4.4:0.0459 0.1121 0.08050 0.15778 0.1636 0.2466 0.8873 0.8978: 0.0463 0.1128 0.0807 0.1583 0.1642 0.2474 0.8874  0.89858
    #4.6:0.0465  0.11296 0.08097  0.15855 0.16493 0.2480 0.8839 0.89637
    #4.8