import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.nn import CosineSimilarity
import math


class MCLGAT(nn.Module):
    """
    MCLGAT: A model combining Multi-View Contrastive Learning and graph attention network (GAT)
    """

    def __init__(self,input_dim,hidden1_dim,hidden2_dim,output_dim,num_head1,num_head2,
                 alpha,device,type,reduction,num_nodes):
        super(MCLGAT, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction
        self.num_nodes=num_nodes
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1*hidden1_dim
            self.hidden2_dim = num_head2*hidden2_dim


        self.ConvLayer1 = [AttentionLayer(input_dim,hidden1_dim,num_nodes,alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module('ConvLayer1_AttentionHead{}'.format(i),attention)

        self.ConvLayer2 = [AttentionLayer(self.hidden1_dim,hidden2_dim,num_nodes,alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i),attention)

        self.tf_linear1 = nn.Linear(384,output_dim)
        self.target_linear1 = nn.Linear(384,output_dim)

        if self.type == 'MLP':
            self.linear = nn.Linear(2*output_dim, 2)

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()

        for attention in self.ConvLayer2:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight,gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)



    def encode(self,x,adj):
        if self.reduction =='concate':

            x = torch.cat([att(x, adj,1)for att in self.ConvLayer1], dim=1)
            x = F.elu(x)


        elif self.reduction =='mean':
            x = torch.mean(torch.stack([att(x, adj,1) for att in self.ConvLayer1]), dim=0)
            x = F.elu(x)

        else:
            raise TypeError


        out = torch.mean(torch.stack([att(x, adj,2) for att in self.ConvLayer2]),dim=0)
        out=F.elu(out)

        return out


    def decode(self,tf_embed,target_embed):

        if self.type =='dot':

            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob,dim=1).view(-1,1)


            return prob

        elif self.type =='cosine':
            prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)

            return prob

        elif self.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)

            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))


    def forward(self,x,adj,train_sample):

        contrastive_losses = []

        if self.reduction == 'concate':
            embeds, loss1 = zip(*[att(x, adj, 1) for att in self.ConvLayer1])
            embed = torch.cat(embeds, dim=1)
        else:
            embeds, loss1 = zip(*[att(x, adj, 1) for att in self.ConvLayer1])
            embed = torch.mean(torch.stack(embeds), dim=0)

        contrastive_losses.extend(loss1)

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.elu(tf_embed)
        tf_embed = F.dropout(tf_embed,p=0.1)
        target_embed = self.target_linear1(embed)
        target_embed = F.elu(target_embed)
        target_embed = F.dropout(target_embed, p=0.1)
        self.tf_ouput = tf_embed
        self.target_output = target_embed


        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        total_loss = sum(contrastive_losses)

        return pred, total_loss

    def get_embedding(self):
        return self.tf_ouput, self.target_output



class AttentionLayer(nn.Module):
    """
    Graph Attention Layer
    """
    def __init__(self,input_dim,output_dim,nums,alpha=0.2,bias=True, tau=0.5, device='cuda:0'):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.num=nums
        self.tau = tau
        self.device = device

        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight2 = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim,self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim,1)))
        self.a2 = nn.Parameter(torch.zeros(size=(2*self.output_dim, 1)))
        self.W = nn.Parameter(torch.FloatTensor(self.output_dim, self.output_dim))
        self.Q = nn.Parameter(torch.FloatTensor(self.output_dim, 1))
        self.v = nn.Parameter(torch.empty(2, 1))
        self.device = device

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight2.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.Q.data, gain=1.414)
        nn.init.xavier_uniform_(self.v.data, gain=1.414)
        nn.init.constant_(self.bias, 0)

    def _prepare_attentional_mechanism_input(self, x,x2):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        Wh3 = torch.matmul(x2, self.a2[:self.output_dim, :])
        Wh4 = torch.matmul(x2, self.a2[self.output_dim:, :])
        e = torch.exp(-torch.square(Wh1 - Wh2.T)/1e-0)
        qk_concat = torch.cat([Wh3, Wh4], dim=-1)
        attention_scores = torch.matmul(qk_concat, self.v)
        e2 = F.leaky_relu(attention_scores, negative_slope=self.alpha)

        return e,e2

    def contrastive_loss(self, view1, view2):

        view1 = F.normalize(view1, p=2, dim=1)
        view2 = F.normalize(view2, p=2, dim=1)

        pos_sim = torch.sum(view1 * view2, dim=1)
        pos_sim = torch.exp(pos_sim / self.tau)

        neg_sim1 = torch.mm(view1, view2.T)
        neg_sim2 = torch.mm(view2, view1.T)

        neg_sim = torch.exp(neg_sim1 / self.tau) + torch.exp(neg_sim2 / self.tau)

        loss = -torch.log(pos_sim / (neg_sim.sum(dim=1) - pos_sim + 1e-8))
        return loss.mean()

    def forward(self,x,adj,layer):

        h=torch.matmul(x,self.weight)
        h2=torch.matmul(x, self.weight2)
        e,e2 = self._prepare_attentional_mechanism_input(h,h2)
        zero_vec = -9e15 * torch.ones_like(e)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        A = adj.to_dense().to(device)
        I = torch.eye(A.shape[0]).to(device)
        A = A + I

        degree = torch.sum(A, dim=1).view(-1, 1)
        RA = torch.mm(A, A) / (degree.T + 1e-10)
        RA = torch.where(torch.isnan(RA), torch.full_like(RA, 0), RA)

        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention2 = torch.where(RA > 0.2, e2, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention2 = F.softmax(attention2, dim=1)
        attention = F.dropout(attention, training=self.training)
        attention2 = F.dropout(attention2, training=self.training)
        h_pass = torch.matmul(attention, h)
        h_pass2=torch.matmul(attention2,h2)

        output_data = h_pass
        output_data2=h_pass2
        output_data = F.leaky_relu(output_data,negative_slope=self.alpha)
        output_data2 = F.leaky_relu(output_data2,negative_slope=self.alpha)

        output_data = F.normalize(output_data,p=2,dim=1)
        output_data2 = F.normalize(output_data2, p=2, dim=1)

        contrastive_loss = self.contrastive_loss(output_data, output_data2)

        w1 = torch.mean(torch.matmul(torch.tanh(torch.matmul( output_data, self.W)), self.Q), dim=0)
        w2 = torch.mean(torch.matmul(torch.tanh(torch.matmul( output_data2, self.W)), self.Q), dim=0)
        w = torch.cat([w1, w2], dim=0)
        w = F.softmax(w, dim=0)
        output_data = w[0] * output_data + w[1] * output_data2




        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data, contrastive_loss












