

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from GNN import GNNLayer



"""
type_hyperedge  不同种类超边数目
num_layers     超边矩阵相乘次数

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class HGTN(nn.Module):
    def __init__(self, type_hyperedge, num_channels,w_in, w_out,num_layers):
        super(HGTN, self).__init__()
        self.type_hyperedge = type_hyperedge
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(HGTLayer(type_hyperedge, num_channels, first=True))
            else:
                layers.append(HGTLayer(type_hyperedge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def hgcn_conv(self,X,H):
        X = torch.mm(X, self.weight)
        return torch.mm(H.t(),X)

    def forward(self, A, X):
        A = A.unsqueeze(0).permute(0,3,1,2) 
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)

            else:
                H, W = self.layers[i](A, H)
            Ws.append(W)
        #print(Ws)
        A = A.squeeze(0).permute(1,2,0) 
        return  A,H,W
        #X特征
        #H 超图



        # #多通道
        # for i in range(self.num_channels):
        #     if i==0:
        #         X_ = F.relu(self.hgcn_conv(X,H[i]))
        #     else:
        #         X_tmp = F.relu(self.hgcn_conv(X,H[i]))
        #         X_ = torch.cat((X_,X_tmp), dim=1)
        # X_ = self.linear1(X_)
        # X_ = F.relu(X_)
        # y = self.linear2(X_[target_x])
        # loss = self.loss(y, target)
        # return loss, y, Ws, H





class HGTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(HGTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = HGNN_Conv(in_channels, out_channels)
            self.conv2 = HGNN_Conv(in_channels, out_channels)
        else:
            self.conv1 = HGNN_Conv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)

            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W


class HGNN_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(HGNN_Conv, self).__init__()

        self.weight = Parameter(torch.Tensor(out_channels,in_channels,1,1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.1)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, A):
        Q = torch.sum(A*F.softmax(self.weight, dim=1),dim=1)
        return Q

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        # extracted feature by AE
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)
    def forward(self, x):
        enc_z2 = F.relu(self.enc_1(x))
        enc_z3 = F.relu(self.enc_2(enc_z2))
        enc_z4 = F.relu(self.enc_3(enc_z3))
        z = self.z_layer(enc_z4)
        dec_z2 = F.relu(self.dec_1(z))
        dec_z3 = F.relu(self.dec_2(dec_z2))
        dec_z4 = F.relu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)

        return x_bar, enc_z2, enc_z3, enc_z4, z

class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        
        return weight_output

class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp,2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1) 
        
        return weight_output

class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)
        
        return weight_output

class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)  
        
        return weight_output
# IGAE encoder from DFCN
class IGAE_encoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z_1 = self.gnn_1(x, adj, active=True)
        z_2 = self.gnn_2(z_1, adj, active=True)
        z_igae = self.gnn_3(z_2, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


# IGAE decoder from DFCN
class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z_1 = self.gnn_4(z_igae, adj, active=True)
        z_2 = self.gnn_5(z_1, adj, active=True)
        z_hat = self.gnn_6(z_2, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


# Improved Graph Auto Encoder from DFCN
class IGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        # IGAE encoder
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        # IGAE decoder
        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)
class SHCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, type_hyperedge, 
                num_channels, w_in, w_out, num_layers, pretrain_path,v=1):
        super(SHCN, self).__init__()

        # AE
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        
        self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))
        
        self.gae = IGAE(
            gae_n_enc_1=n_enc_1,
            gae_n_enc_2=n_enc_2,
            gae_n_enc_3=n_enc_3,
            gae_n_dec_1=n_dec_1,
            gae_n_dec_2=n_dec_2,
            gae_n_dec_3=n_dec_3,
            n_input= n_input)
        self.hgtn = HGTN(

                type_hyperedge=type_hyperedge, 
                num_channels=num_channels, 
                w_in=w_in, 
                w_out=w_out, 
                num_layers=num_layers)

        self.agcn_0 = GNNLayer(n_input, n_enc_1)
        self.agcn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.agcn_2 = GNNLayer(n_enc_2, n_enc_3)
        self.agcn_3 = GNNLayer(n_enc_3, n_z)
        self.agcn_z = GNNLayer(3020,n_clusters)

        self.mlp = MLP_L(3020)

        # attention on [Z_i || H_i]
        self.mlp1 = MLP_1(2*n_enc_1)
        self.mlp2 = MLP_2(2*n_enc_2)
        self.mlp3 = MLP_3(2*n_enc_3)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, X, H):
        # AE Module
        x_bar, h1, h2, h3, z = self.ae(X)

        x_array = list(np.shape(X))
        n_x = x_array[0]

        # # AGCN-H
        z1 = self.agcn_0(X,H)
        # z2

        z2 = self.agcn_1( z1, H)
        # z3

        z3 = self.agcn_2( z2, H)
        # z4

        z4 = self.agcn_3( z3, H)
        Z_hat, Z_adj_hat = self.gae.decoder(z3, H)
        # # AGCN-S


        net_output = torch.cat((z1, z2, z3, z4, z), 1 )   
        net_output = self.agcn_z(net_output, H, active=False) 
        predict = F.softmax(net_output, dim=1)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, net_output,Z_hat,Z_adj_hat,H

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()