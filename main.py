

import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
from model import SHCN, target_distribution
import hg_con as hgcon
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import f1_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
from sklearn.decomposition import PCA
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from evaluation import eva
import time

from sklearn.cluster import KMeans
from pytorch_metric_learning import losses
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='DBLP',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=128,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=1,
                        help='number of channels')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layer')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='l2 reg')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    parser.add_argument('--variable_weight',default=False)
    parser.add_argument('--n_clusters', type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    num_layers = args.num_layers
    lr = args.lr
    weight_decay = args.weight_decay
    norm = args.norm
    adaptive_lr = args.adaptive_lr
    dataname = args.dataset
    variable_weight = args.variable_weight
    #n_clusters = args.n_clusters
    n_z = args.n_z
    
    if dataname == 'DBLP':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph_path = 'data/{}/{}_graph'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_k_3_graph'.format(dataname,dataname)
        #data_graph2_path = 'data/{}/{}_k_5_graph'.format(dataname,dataname)
        #data_graph3_path = 'data/{}/{}_k_10_graph'.format(dataname,dataname)
        #data_graph4_path = 'data/{}/{}_k_20_graph'.format(dataname,dataname)

        features = np.loadtxt(data_features_path+'.txt')
        #features = np.loadtxt(data_features_path+'.txt')
        #pca = PCA(n_components=50)
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph = np.loadtxt(data_graph_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        # graph2 = np.loadtxt(data_graph2_path+'.txt') 
        # graph3 = np.loadtxt(data_graph3_path+'.txt')
        #graph4 = np.loadtxt(data_graph3_path+'.txt')
        pretrain_path = 'data/{}/{}.pkl'.format(dataname,dataname)
        args.num_layers = 2
        n_clusters = 4
        lr=0.001
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph, graph1]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            #G = hgcon.load_graph(features, graph)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        #print(A)
        #print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')

    
    if dataname == 'ACM':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph_path = 'data/{}/{}_graph'.format(dataname,dataname)
        # data_graph1_path = 'data/{}/{}_PAP'.format(dataname,dataname)
        # data_graph2_path = 'data/{}/{}_PLP'.format(dataname,dataname)
        #data_graph3_path = 'data/{}/{}_PMP'.format(dataname,dataname)
        #data_graph4_path = 'data/{}/{}_PTP'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #pca = PCA(n_components=100)
        #features = pca.fit_transform(features)
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph = np.loadtxt(data_graph_path+'.txt') 
        data_graph1_path = 'data/{}/{}_k_3_graph'.format(dataname,dataname)
        #data_graph2_path = 'data/{}/{}_k_5_graph'.format(dataname,dataname)
        #data_graph3_path = 'data/{}/{}_k_10_graph'.format(dataname,dataname)
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        # graph2 = np.loadtxt(data_graph2_path+'.txt') 
        #graph3 = np.loadtxt(data_graph3_path+'.txt')
        #graph4 = np.loadtxt(data_graph4_path+'.txt')
        pretrain_path = 'data/{}/{}.pkl'.format(dataname,dataname)
        args.num_layers = 2
        n_clusters = 3
        lr=0.001
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph,graph1]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            #G = hgcon.load_graph(features, graph)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        #print(A)
        #print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')

        
    if dataname == 'CITE':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_graph'.format(dataname,dataname)
        data_graph2_path = 'data/{}/{}_k_5_graph'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        graph2 = np.loadtxt(data_graph2_path+'.txt') 
        pretrain_path = 'data/{}/{}.pkl'.format(dataname,dataname)
        args.num_layers = 2
        n_clusters = 6
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph1,graph2]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            G = hgcon.load_graph(features, graph)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')


    if dataname == 'REUT':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_graph5'.format(dataname,dataname)
        data_graph2_path = 'data/{}/{}_graph10'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        graph2 = np.loadtxt(data_graph2_path+'.txt') 
        pretrain_path = 'data/{}/{}.pkl'.format(dataname,dataname)
        args.num_layers = 2
        n_clusters = 4
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph1,graph2]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            G = hgcon.load_graph(features, graph)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')
    

    if dataname == 'USPS':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_graph1'.format(dataname,dataname)
        data_graph2_path = 'data/{}/{}_graph3'.format(dataname,dataname)
        data_graph3_path = 'data/{}/{}_graph5'.format(dataname,dataname)
        data_graph4_path = 'data/{}/{}_graph10'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        graph2 = np.loadtxt(data_graph2_path+'.txt') 
        graph3 = np.loadtxt(data_graph3_path+'.txt') 
        graph4 = np.loadtxt(data_graph4_path+'.txt') 
        pretrain_path = 'data/{}/{}.pkl'.format(dataname,dataname)
        args.num_layers = 3
        n_clusters = 10
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph2,graph3]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            G = hgcon.load_graph(features, graph)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')
    
    if dataname == 'HHAR':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_graph1'.format(dataname,dataname)
        data_graph2_path = 'data/{}/{}_graph3'.format(dataname,dataname)
        data_graph3_path = 'data/{}/{}_graph5'.format(dataname,dataname)
        data_graph4_path = 'data/{}/{}_graph10'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        graph2 = np.loadtxt(data_graph2_path+'.txt') 
        graph3 = np.loadtxt(data_graph3_path+'.txt') 
        graph4 = np.loadtxt(data_graph4_path+'.txt') 
        pretrain_path = 'data/{}/{}.pkl'.format(dataname,dataname)
        args.num_layers = 3
        n_clusters = 6
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph1,graph2]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            G = hgcon.load_graph(features, graph)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        #print(A)
        #print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')     
            
    eprm_state = 'result'

    file_out = open('./output/'+dataname+'_'+eprm_state+'.txt', 'a')
    print("The experimental results", file=file_out)

    # hyper parameters
    lambda_1 = [1] #[0.01,0.1,1,10,100]
    lambda_2 = [1] #[0.01,0.1,1,10,100]
    #lambda_1 = [0.01,0.1,1,10,100,1000]
    #lambda_2 = [0.01,0.1,1,10,100,1000]
    kmeans_max = 0
    for ld1 in lambda_1:
        for ld2 in lambda_2:
            print("lambda_1: ", ld1, "lambda_2: ", ld2, file=file_out)
            # model = AGCN(500, 500, 2000, 2000, 500, 500,
            #             n_input=args.n_input,
            #             n_z=args.n_z,
            #             n_clusters=args.n_clusters,
            #             v=1.0).cuda()


            model = SHCN(500, 500, 2000, 2000, 500, 500,
                        n_input=features.shape[1],
                        n_z=n_z,
                        n_clusters=n_clusters,
                        pretrain_path=pretrain_path, 
                        type_hyperedge=A.shape[-1], 
                        num_channels=num_channels, 
                        w_in = features.shape[1], 
                        w_out = node_dim,
                        num_layers=num_layers,
                        v=1.0)
            model = model.to(device)

            optimizer = Adam(model.parameters(), lr=args.lr)

            # KNN Graph
            adj_m = A.to(device)
            #adj = adj.cuda()

            # cluster parameter initiate
            #data = torch.Tensor(dataset.x).cuda()
            data = torch.Tensor(features).to(device)
            y = labels
            #adj,data = model.hgtn(adj,data)
            with torch.no_grad():
                _, _, _, _, z = model.ae(data)
                z = z.to(device)
            iters10_kmeans_iter_Q = []
            iters10_NMI_iter_Q = []
            iters10_ARI_iter_Q = []
            iters10_F1_iter_Q = []

            iters10_kmeans_iter_Z = []
            iters10_NMI_iter_Z = []
            iters10_ARI_iter_Z = []
            iters10_F1_iter_Z = []

            iters10_kmeans_iter_P = []
            iters10_NMI_iter_P = []
            iters10_ARI_iter_P = []
            iters10_F1_iter_P = []

            z_1st = z

            for i in range(1):

                kmeans = KMeans(n_clusters=n_clusters, n_init=20)
                y_pred = kmeans.fit_predict(z_1st.data.cpu().numpy())
                y_pred_last = y_pred
                #model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()
                model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
                acc,nmi,ari,f1 = eva(y, y_pred, 'pae')

                # get the value
                kmeans_iter_Q = []
                NMI_iter_Q = []
                ARI_iter_Q = []
                F1_iter_Q = []

                kmeans_iter_Z = []
                NMI_iter_Z = []
                ARI_iter_Z = []
                F1_iter_Z = []

                kmeans_iter_P = []
                NMI_iter_P = []
                ARI_iter_P = []
                F1_iter_P = []
                
                Ws = np.zeros(shape=[200,3])
                for epoch in range(epochs):

                    if epoch % 1 == 0:
                        #print(1)
                        #print(adj_m.shape)
 
                        adj_m,adj,W = model.hgtn(adj_m,data)

                        #Ws0 = (W[1].data.cpu().numpy())[0].ravel()[0]
                        #Ws1 = (W[1].data.cpu().numpy())[0].ravel()[1]
                        #Ws2 = (W[1].data.cpu().numpy())[0].ravel()[2]
                        #Ws[epoch,0] = Ws0
                        #Ws[epoch,1] = Ws1
                        #Ws[epoch,2] = Ws2
                        
                        
                        #print(2)
                        #print(adj.shape)
                        #adj = adj.detach().numpy()
                        adj=adj.squeeze(0)
                        #print(3)
                        #print(adj.shape)
                        _, tmp_q, pred, z, _,_,_,H = model(data, adj)
                        

                        
                        
                        tmp_q = tmp_q.data
                        p = target_distribution(tmp_q)
                    
                        res1 = tmp_q.cpu().numpy().argmax(1)       #Q
                        res2 = pred.data.cpu().numpy().argmax(1)   #Z
                        #res2 = kmeans.fit_predict(pred.data.cpu().numpy())
                        #print(res2)
                        res3 = p.data.cpu().numpy().argmax(1)      #P

                        acc,nmi,ari,f1 = eva(y, res1, str(epoch) + 'Q')
                        kmeans_iter_Q.append(acc)
                        NMI_iter_Q.append(nmi)
                        ARI_iter_Q.append(ari)
                        F1_iter_Q.append(f1 )



                        acc,nmi,ari,f1 = eva(y, res3, str(epoch) + 'P')
                        kmeans_iter_P.append(acc)
                        NMI_iter_P.append(nmi)
                        ARI_iter_P.append(ari)
                        F1_iter_P.append(f1)
                        
                        acc,nmi,ari,f1 = eva(y, res2, str(epoch) + 'Z')
                        kmeans_iter_Z.append(acc)
                        NMI_iter_Z.append(nmi)
                        ARI_iter_Z.append(ari)
                        F1_iter_Z.append(f1)
                    
                    x_bar, q, pred, _, net_output,Z_hat,A_bar,H = model(data, adj)
                    #adj=adj.unsqueeze(0)
                    kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
                    ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
                    #kl_loss = essloss(q, p)
                    #ce_loss = essloss(pred, p)
                    re_loss = F.mse_loss(x_bar, data) ####

                    #print(kl_loss)
                    #print(ce_loss)
                    loss = ld1 * kl_loss + ld2 * ce_loss 

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # _Z

                kmeans_max= np.max(kmeans_iter_Z)
                nmi_max= np.max(NMI_iter_Z)
                ari_max= np.max(ARI_iter_Z)
                F1_max= np.max(F1_iter_Z)
                iters10_kmeans_iter_Z.append(round(kmeans_max,5))
                iters10_NMI_iter_Z.append(round(nmi_max,5))
                iters10_ARI_iter_Z.append(round(ari_max,5))
                iters10_F1_iter_Z.append(round(F1_max,5))
                if np.max(iters10_kmeans_iter_Z) < acc:
                    np.savetxt(dataname+'.txt',pred.data.cpu().numpy())
                
                
                # _Q
                kmeans_max= np.max(kmeans_iter_Q)
                nmi_max= np.max(NMI_iter_Q)
                ari_max= np.max(ARI_iter_Q)
                F1_max= np.max(F1_iter_Q)
                iters10_kmeans_iter_Q.append(round(kmeans_max,5))
                iters10_NMI_iter_Q.append(round(nmi_max,5))
                iters10_ARI_iter_Q.append(round(ari_max,5))
                iters10_F1_iter_Q.append(round(F1_max,5))


                # _P
                kmeans_max= np.max(kmeans_iter_P)
                nmi_max= np.max(NMI_iter_P)
                ari_max= np.max(ARI_iter_P)
                F1_max= np.max(F1_iter_P)
                #Ws.append((W.tolist()))
                iters10_kmeans_iter_P.append(round(kmeans_max,5))
                iters10_NMI_iter_P.append(round(nmi_max,5))
                iters10_ARI_iter_P.append(round(ari_max,5))
                iters10_F1_iter_P.append(round(F1_max,5))
            #np.savetxt(dataname+'_attention.txt',Ws)
            #np.savetxt(dataname+'_H.txt',H.data.cpu().numpy())
            #np.savetxt(dataname+'.txt',net_output.data.cpu().numpy())
            print("#####################################", file=file_out)
            print(args, file=file_out)
            print("kmeans Z mean",round(np.mean(iters10_kmeans_iter_Z),5),"max",np.max(iters10_kmeans_iter_Z),"\n",iters10_kmeans_iter_Z, file=file_out)
            print("NMI mean",round(np.mean(iters10_NMI_iter_Z),5),"max",np.max(iters10_NMI_iter_Z),"\n",iters10_NMI_iter_Z, file=file_out)
            print("ARI mean",round(np.mean(iters10_ARI_iter_Z),5),"max",np.max(iters10_ARI_iter_Z),"\n",iters10_ARI_iter_Z, file=file_out)
            print("F1  mean",round(np.mean(iters10_F1_iter_Z),5),"max",np.max(iters10_F1_iter_Z),"\n",iters10_F1_iter_Z, file=file_out)
            print(':acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(round(np.mean(iters10_kmeans_iter_Z),5),round(np.mean(iters10_NMI_iter_Z),5),round(np.mean(iters10_ARI_iter_Z),5),round(np.mean(iters10_F1_iter_Z),5)), file=file_out)

    #file_out.close()

