import torch
import torch.nn as nn
from model_lstm.AGCN import AVWGCN

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 3*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)
        self.space = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, y,state, cell,last_state,node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        cell = cell.to(x.device)
        last_state = last_state.to(y.device)
        input_and_state = torch.cat((x, state), dim=-1)
        input_and_last_state = torch.cat((y, last_state), dim=-1)
        
        
        f_i_o = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        last_state = torch.sigmoid(self.space(input_and_last_state, node_embeddings))
        f,i,o = torch.split(f_i_o, self.hidden_dim, dim=-1)
        g = torch.tanh(self.update(input_and_state, node_embeddings))
        c=f*cell+i*g
        h = o*torch.tanh(c)
        return h,c,last_state

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    