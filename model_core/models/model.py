import torch
from torch import nn
from opt_einsum import contract
from config import *
import os
import time
from typing import Optional, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, AutoConfig
from models.graph import Directed_Bipartite_Graph_Encoder
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Co_occurrence_Infused_Multi_Label_Attention(nn.Module):
    def __init__(self, d_model, transform_size, graph_att_hidd, graph_ffn_hidd, n_heads, ground_ind_tail_file, 
                 ground_ind_head_file, adj_matrix_file, c_indices_file):
        super(Co_occurrence_Infused_Multi_Label_Attention, self).__init__()

        self.d_k = int(transform_size /n_heads)
        self.d_v = self.d_k
        self.n_heads = n_heads

        self.trans = nn.Linear(d_model, self.d_k * n_heads)
        self.Q_proj = nn.Linear(self.d_k * n_heads, self.d_k * n_heads)
        self.K_proj = nn.Linear(d_model, self.d_k * n_heads)
        self.V_proj = nn.Linear(d_model, self.d_k * n_heads)

        self.layernorm = nn.LayerNorm(self.d_k * n_heads)
        self.W = nn.Linear(self.d_k * n_heads, self.d_k * n_heads, bias=False)
        self.ground_ind_tail_file = ground_ind_tail_file
        self.ground_ind_head_file = ground_ind_head_file

        self.adj_matrix_file = adj_matrix_file
        self.c_indices_file = c_indices_file

        # load comorbidity encoding indices
        self.ground_ind_tail = list(pickle.load(open(ground_ind_tail_file, "rb")).values())
        self.ground_ind_head = list(pickle.load(open(ground_ind_head_file, "rb")).values())
        self.adj_matrix = torch.tensor(pickle.load(open(adj_matrix_file, "rb")),dtype=torch.int64).to(DEVICE)
        self.c_indices = torch.tensor(pickle.load(open(c_indices_file, "rb"))).to(DEVICE)

        self.graph_att_hidd = graph_att_hidd
        self.graph_ffn_hidd = graph_ffn_hidd

        if MODE == "train":
            self.graph_encoder = Directed_Bipartite_Graph_Encoder(
                num_encoder_layers=GRAPH_NUM,
                att_hidd=self.graph_att_hidd,
                ffn_hidd=self.graph_ffn_hidd,
                num_attention_heads=NUM_ATT_HEAD,
                dropout=0.1,
                c_indices=self.c_indices,
                ground_ind_tail=self.ground_ind_tail,
                ground_ind_head=self.ground_ind_head,
            ).to(torch.bfloat16).to(DEVICE)
        else:
            self.graph_encoder = Directed_Bipartite_Graph_Encoder(
                num_encoder_layers=GRAPH_NUM,
                att_hidd=self.graph_att_hidd,
                ffn_hidd=self.graph_ffn_hidd,
                num_attention_heads=NUM_ATT_HEAD,
                dropout=0.1,
                c_indices=self.c_indices,
                ground_ind_tail=self.ground_ind_tail,
                ground_ind_head=self.ground_ind_head,
            ).to(DEVICE)


    def forward(self, Q, H, a):
        # build Qg for graph encoder
        
        Qg = self.trans(Q)
        Qg = nn.Tanh()(Qg)
        Qg = self.graph_encoder(Qg, self.adj_matrix)

        n_classes = Qg.size(0)
        Chunk_num = H.size(0)

        q = self.Q_proj(Qg).view(n_classes, self.n_heads, self.d_k)
        k = self.K_proj(H).view(Chunk_num, -1, self.n_heads, self.d_k).transpose(1,2)
        v = self.V_proj(H).view(Chunk_num, -1, self.n_heads, self.d_k).transpose(1,2)

        context = DotProductAttention(self.d_k)(q, k, v,a)

        context = context.reshape(Chunk_num,n_classes,self.n_heads * self.d_v)

        Qg = self.W(Qg)

        output = contract("bch,ch->bc",context,Qg)

        return output

class DotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(DotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, WQ, WK,WV,a):
        WK = nn.Tanh()(WK)
        WV = nn.Tanh()(WV)
        scores = contract('bzth,czh->bczt',WK, WQ)
        num_chunks, num_codes, n_heads, token_lenth = scores.size()
        scores = scores.to(DEVICE)
        scores = scores.view(num_chunks, num_codes, n_heads, token_lenth)
        a = a.unsqueeze(1).unsqueeze(2)
        attn_mask = a.expand(num_chunks, num_codes, n_heads, token_lenth)
        attn = nn.Softmax(dim=-1)(scores+attn_mask)

        context = contract('bzth,bczt->bczh',WV,attn)

        return context


class ProBias(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self,config):
        super(ProBias, self).__init__(config)

        model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels).to(DEVICE)
        self.model = model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        if not os.path.exists(OUTPUT_DIR + "/{}_embeddings.pkl".format(CODE_TYPE)):
                    
            with open(config.code_token_file, 'rb') as f:
                self.code_token = pickle.load(f).to(DEVICE)
            self.out_code = []
            dataset = TensorDataset(
                self.code_token["input_ids"],
                self.code_token["token_type_ids"],
                self.code_token["attention_mask"]
            )
            self.model.eval()
            with torch.no_grad():
                label_token_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
                
                for input_ids_batch, token_type_ids_batch, attention_mask_batch in label_token_dataloader:
                    out_code = self.model.bert(
                        input_ids=input_ids_batch,
                        token_type_ids=token_type_ids_batch,
                        attention_mask=attention_mask_batch,
                        return_dict = False
                    )[0]
                    self.out_code.append(out_code[:,0,:])
            self.out_code = torch.cat(self.out_code, dim=0)
            with open(OUTPUT_DIR + "/{}_embeddings.pkl".format(CODE_TYPE), 'wb') as f:
                pickle.dump(self.out_code, f)
        else:
            with open(OUTPUT_DIR + "/{}_embeddings.pkl".format(CODE_TYPE), 'rb') as f:
                self.out_code = pickle.load(f)
        self.attention = Co_occurrence_Infused_Multi_Label_Attention(config.attention_hidden_size,config.transform_size, config.graph_att_hidd, config.graph_ffn_hidd, config.n_heads,
                                                     config.ground_ind_tail_file, config.ground_ind_head_file,config.adj_matrix_file, config.c_indices_file)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None):
        encoder_outputs = self.model.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_dict = False
            )
        sentences_encodings = encoder_outputs[0]
        codes = self.out_code #Codes
        documents = sentences_encodings #Documents
        classification_outputs = self.attention(codes,documents,attention_mask)
        predictions = nn.MaxPool1d(classification_outputs.shape[0], stride=classification_outputs.shape[0])(classification_outputs.permute(1,0)).T

        return predictions

class Model(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self,config):
        super(Model,self).__init__(config)

        self.model = ProBias(config)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None):
        
        return self.model(input_ids,token_type_ids,attention_mask,labels)
