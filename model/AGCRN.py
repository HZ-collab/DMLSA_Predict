import torch
import torch.nn as nn
from model.AGCRNCell import AGCRNCell
import math
class TransformerLayer(nn.Module):
    def __init__(self, d_model,num_head, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # self.attan=nn.TransformerEncoderLayer(d_model,num_head,dim_feedforward=d_model*2,dropout=dropout)
        self.attan=nn.MultiheadAttention(d_model,num_head,dropout,bias=True)
  
    def forward(self, X,K=None,V=None):
        batch_size, seq_len, num_nodes, num_feat  = X.shape
        X=X.transpose(0,1).reshape(seq_len,batch_size*num_nodes,num_feat)
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        # X=self.attan(X)
        X=self.attan(X,X,X)[0]
        X=X.reshape(seq_len,batch_size,num_nodes,num_feat)
        X=X.transpose(0,1)
        return X


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1,num_head=4,dropout=0.1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.translayers = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        self.translayers.append(TransformerLayer(dim_out,num_head,dropout))
        self.dcrnn_cell1__weight_mean = None
        self.dcrnn_cell1__weight_fisher = None
        self.dcrnn_cell1__bias_mean = None
        self.dcrnn_cell1__bias_fisher = None

        self.translayer1__weight_mean = None
        self.translayer1__weight_fisher = None
        self.translayer1__bias_mean = None
        self.translayer1__bias_fisher = None

        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))
            self.translayers.append(TransformerLayer(dim_out,num_head,dropout))

        

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        # print(x.shape[2],self.node_num,x.shape[3],self.input_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        current_last_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            last_state=init_state[i]
            inner_states = []
            inner_last_states = []
            for t in range(seq_length):
                state,last_state = self.dcrnn_cells[i](current_inputs[:, t, :, :],current_last_inputs[:, t, :, :], state, last_state,node_embeddings)
                inner_states.append(state)
                inner_last_states.append(last_state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
            current_last_inputs = torch.stack(inner_last_states, dim=1)
            # current_inputs = torch.stack(inner_states, dim=1)   
            current_inputs=self.translayers[i](current_inputs)  #打开表示为改进后的方法
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs_short = x
        current_last_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            last_state=init_state[i]
            inner_states = []
            inner_last_states = []
            for t in range(seq_length):
                state,last_state = self.dcrnn_cells[i](current_inputs_short[:, t, :, :],current_last_inputs[:, t, :, :], state, last_state,node_embeddings)
                inner_states.append(state)
                inner_last_states.append(last_state)
            output_hidden.append(state)
            current_inputs_short = torch.stack(inner_states, dim=1)
            current_last_inputs = torch.stack(inner_last_states, dim=1)
            # current_inputs = torch.stack(inner_states, dim=1)   
            # current_inputs=self.translayers[i](current_inputs)  #打开表示为改进后的方法
        
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden,current_last_inputs,current_inputs_short

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)
    

class ChannelWeighting3(nn.Module):
    def __init__(self, num_channels):
        super(ChannelWeighting3, self).__init__()
        # 定义一个可学习的权重参数，为每个通道设置初始权重
        # 注意：这里使用nn.Parameter来确保这些权重会被认为是模型参数
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def forward(self, x):
        # 检查输入数据的通道数是否与权重匹配
        assert x.size(1) == self.weights.size(0), "输入通道数与权重不匹配"
        # 将权重调整为与输入数据的形状匹配，以便进行逐元素乘法
        weights = self.weights.view(1, -1, 1, 1)
        # 对输入数据的每个通道应用权重
        return x * weights
    
class ChannelWeighting2(nn.Module):
    def __init__(self, num_channels):
        super(ChannelWeighting2, self).__init__()
        # 定义一个可学习的权重参数，为每个通道设置初始权重
        # 注意：这里使用nn.Parameter来确保这些权重会被认为是模型参数
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward(self, x):
        # 检查输入数据的通道数是否与权重匹配
        assert x.size(1) == self.weights.size(0), "输入通道数与权重不匹配"
        # 将权重调整为与输入数据的形状匹配，以便进行逐元素乘法
        weights = self.weights.view(1, -1, 1, 1)
        # 对输入数据的每个通道应用权重
        return x * weights

class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.type=args.type
        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers,args.num_head,args.dropout)
        for n, p in self.encoder.named_parameters():
            attr_name1 = '{}_mean'.format(n)
            attr_name2 = '{}_fisher'.format(n)
            setattr(self.encoder, attr_name1, None)
            setattr(self.encoder, attr_name2, None)

        #predictor
        if self.type=='all':
            self.channel_weighting = ChannelWeighting3(num_channels=3)
            self.end_conv = nn.Conv2d(3, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        else:
            self.channel_weighting = ChannelWeighting2(num_channels=3)
            self.end_conv = nn.Conv2d(2, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])  #初始化层，（层数、批量、节点、维度）
        output, _,last_state,current_inputs_short= self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        last_state = last_state[:, -1:, :, :]                                   #B, 1, N, hidden
        current_inputs_short = current_inputs_short[:, -1:, :, :]                                   #B, 1, N, hidden
        if self.type=='all':
            output = torch.cat((output, last_state, current_inputs_short), dim=1)
        elif self.type=='offSpace':
            output = torch.cat((output, current_inputs_short), dim=1)
        elif self.type=='offShort':
            output = torch.cat((output, last_state), dim=1)
        elif self.type=='offLong':
            output = torch.cat((last_state, current_inputs_short), dim=1)
        else:
            return None

 

        #CNN based predictor
        output = self.channel_weighting(output)
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output
    