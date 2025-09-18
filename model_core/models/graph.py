import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config import *

class Directed_Bipartite_MH_Attention(nn.Module):
    def __init__(self, att_hidd_dim=None, num_heads=None, dropout=None, c_indices=None, ground_ind_tail=None, ground_ind_head=None):
        super().__init__()
        self.att_hidd_dim = att_hidd_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.head_dim = att_hidd_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.c_indices = c_indices
        self.edge_encoder = nn.Embedding(self.c_indices.flatten().max()+1, num_heads)
        self.ground_ind_tail = torch.tensor(ground_ind_tail).to(DEVICE)
        self.ground_ind_head = torch.tensor(ground_ind_head).to(DEVICE)
        
        self.q_proj = nn.Linear(att_hidd_dim, att_hidd_dim)
        self.k_proj = nn.Linear(att_hidd_dim, att_hidd_dim)
        self.v_proj = nn.Linear(att_hidd_dim, att_hidd_dim)

        self.out_proj = nn.Linear(att_hidd_dim, att_hidd_dim)

    def forward(self, query=None, key=None, value=None, adj_matrix=None):
        q_tail = (self.q_proj(query[self.ground_ind_tail]) * self.scaling).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k_head = self.k_proj(key[self.ground_ind_head]).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v_head = self.v_proj(value[self.ground_ind_head]).view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        tail_mask = torch.where(adj_matrix == 0, -1e9, 0.0).unsqueeze(0)
        edge_bias = self.edge_encoder(self.c_indices).permute(2, 0, 1)

        attn_values_all = torch.zeros(self.num_heads, len(query), self.head_dim, dtype=query.dtype, device=DEVICE)
        if BIAS_TYPE == "N":
            if MODE == "train":
                attn_values_all[:, self.ground_ind_tail,:] = (F.softmax(torch.bmm(q_tail, k_head.transpose(1, 2))+tail_mask, dim=-1).to(torch.bfloat16)) @ v_head
            else:
                attn_values_all[:, self.ground_ind_tail,:] = (F.softmax(torch.bmm(q_tail, k_head.transpose(1, 2))+tail_mask, dim=-1)) @ v_head
        else:
            if MODE =="train":
                attn_values_all[:, self.ground_ind_tail,:] = (F.softmax(torch.bmm(q_tail, k_head.transpose(1, 2))+tail_mask+edge_bias, dim=-1).to(torch.bfloat16)) @ v_head
            else:
                attn_values_all[:, self.ground_ind_tail,:] = (F.softmax(torch.bmm(q_tail, k_head.transpose(1, 2))+tail_mask+edge_bias, dim=-1)) @ v_head
        attn_values_all[:, self.ground_ind_head,:] = query[self.ground_ind_head].view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        attn_values_all = attn_values_all.transpose(0, 1).contiguous().view(-1, self.att_hidd_dim)
        
        return self.out_proj(attn_values_all)


class Directed_Bipartite_Graph_EncoderLayer(nn.Module):
    def __init__(self, att_hidd_dim, ffn_hidd_dim, num_attention_heads, dropout,c_indices,ground_ind_tail,ground_ind_head):
        super().__init__()
        self.self_attn = Directed_Bipartite_MH_Attention(att_hidd_dim=att_hidd_dim, num_heads=num_attention_heads, dropout=dropout,c_indices=c_indices,
                                            ground_ind_tail=ground_ind_tail,ground_ind_head=ground_ind_head)

        self.self_attn_layer_norm = nn.LayerNorm(att_hidd_dim)
        self.fc1 = nn.Linear(att_hidd_dim, ffn_hidd_dim)
        self.fc2 = nn.Linear(ffn_hidd_dim, att_hidd_dim)
        
        self.final_layer_norm = nn.LayerNorm(att_hidd_dim)
        self.dropout_module = nn.Dropout(dropout)
        self.activation_fn = nn.GELU()

    def forward(self, x, adj_matrix=None):
        x = x + self.dropout_module(self.self_attn(x, x, x, adj_matrix))
        
        if MODE == "train":
            x = self.self_attn_layer_norm(x).to(torch.bfloat16)
        else:
            x = self.self_attn_layer_norm(x)
        
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if MODE == "train":
            x = self.final_layer_norm(x).to(torch.bfloat16)
        else:
            x = self.final_layer_norm(x)

        return x

class Directed_Bipartite_Graph_Encoder(nn.Module):
    def __init__(self, num_encoder_layers=None, att_hidd_dim=None, 
                        ffn_hidd_dim=None, num_attention_heads=None, dropout=None,c_indices=None,ground_ind_tail=None,ground_ind_head=None):
        super().__init__()
        self.layers = nn.ModuleList([
            Directed_Bipartite_Graph_EncoderLayer(att_hidd_dim=att_hidd_dim, ffn_hidd_dim=ffn_hidd_dim, num_attention_heads=num_attention_heads, dropout=dropout,
                                        c_indices=c_indices,ground_ind_tail=ground_ind_tail,ground_ind_head=ground_ind_head)
            for _ in range(num_encoder_layers)
        ])

    def forward(self, x, adj_matrix):

        for layer in self.layers:
            x = layer(x, adj_matrix)

        return x