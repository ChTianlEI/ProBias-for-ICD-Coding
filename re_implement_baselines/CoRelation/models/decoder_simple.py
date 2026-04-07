import torch
from torch import nn
import torch.nn.functional as F
import math
from opt_einsum import contract


class SimpleLAAT(nn.Module):
    """
    Simple Label-Aware Attention decoder.
    Compatible with Gatortron encoder output.
    """
    
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['input_dim']
        self.attention_dim = config.get('attention_dim', 512)
        self.num_labels = config['num_labels']
        
        self.W = nn.Linear(self.input_dim, self.attention_dim)
        self.U = nn.Linear(self.attention_dim, self.num_labels, bias=False)
        self.final = nn.Linear(self.input_dim, self.num_labels)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.U.weight)
    
    def forward(self, h, word_mask, label_feat=None, term_count=1, mlabel_feat=None, return_attentions=False):
        """
        Args:
            h: (batch_size, seq_len, hidden_dim)
            word_mask: (batch_size, seq_len)
            label_feat: optional label features
        
        Returns:
            logits: (batch_size, num_labels)
            alphas: attention weights (optional)
        """
        batch_size, seq_len, _ = h.size()
        
        z = torch.tanh(self.W(h))
        
        if label_feat is not None:
            label_count = label_feat.shape[0] // term_count
            label_feat_pooled = label_feat.view(label_count, term_count, -1).mean(dim=1)
            score = torch.matmul(label_feat_pooled, z.transpose(1, 2))
        else:
            score = self.U(z).transpose(1, 2)
        
        if word_mask is not None:
            mask = word_mask.unsqueeze(1).expand_as(score)
            score = score.masked_fill(~mask.bool(), float('-inf'))
        
        alpha = F.softmax(score, dim=-1)
        
        m = torch.matmul(alpha, h)
        
        logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        
        if return_attentions:
            return logits, alpha, None
        return logits, None


class SimpleCoRelation(nn.Module):
    """
    Simplified CoRelation decoder with multi-head attention.
    """
    
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['input_dim']
        self.attention_dim = config.get('attention_dim', 512)
        self.attention_head = config.get('attention_head', 1)
        self.attention_head_dim = config.get('attention_head_dim', 256)
        self.num_labels = config['num_labels']
        self.text_pooling = config.get('text_pooling', 'max')
        self.head_pooling = config.get('head_pooling', 'mean')
        
        self.W = nn.Linear(self.input_dim, self.attention_head * self.attention_head_dim)
        self.V = nn.Linear(self.input_dim, self.attention_dim, bias=False)
        self.u_reduce = nn.Linear(self.attention_dim, self.attention_head * self.attention_head_dim)
        
        self.w_linear = nn.Linear(self.attention_dim, self.attention_dim)
        self.b_linear = nn.Linear(self.attention_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.u_reduce.weight)
    
    def forward(self, h, word_mask, label_feat=None, term_count=1, indices=None, mlabel_feat=None, return_attentions=False):
        """
        Args:
            h: (batch_size, seq_len, hidden_dim)
            word_mask: (batch_size, seq_len)
            label_feat: (num_labels * term_count, hidden_dim)
        
        Returns:
            scores: (batch_size, num_labels) - sigmoid activated
            alphas: attention weights
        """
        batch_size, seq_len, _ = h.size()
        
        if word_mask is not None:
            l = word_mask.shape[-1]
            h = h[:, :l]
        
        z = self.W(h)
        v = self.V(h)
        
        z_reshape = z.view(batch_size, -1, self.attention_head, self.attention_head_dim)
        v_reshape = v.view(batch_size, -1, self.attention_head, self.attention_dim // self.attention_head)
        
        if label_feat is not None:
            label_count = label_feat.size(0) // term_count
            u = self.u_reduce(label_feat)
            u_reshape = u.view(label_count, term_count, self.attention_head, self.attention_head_dim)
            
            score = contract('blhe,cshe->bcshl', z_reshape, u_reshape)
        else:
            label_count = self.num_labels
            score = torch.zeros(batch_size, label_count, 1, self.attention_head, seq_len, device=h.device)
        
        score = score / math.sqrt(self.attention_head_dim)
        
        if word_mask is not None:
            mask = word_mask[:, :score.shape[-1]].bool()
            mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(score)
            score = score.masked_fill(~mask, float('-1e4'))
        
        alpha = F.softmax(score, dim=-1)
        
        m = contract('blhe,bcshl->bcshe', v_reshape, alpha)
        m = m.view(batch_size, label_count, term_count, -1)
        
        if self.text_pooling == 'max':
            m = m.max(dim=2)[0]
        elif self.text_pooling == 'mean':
            m = m.mean(dim=2)
        
        m = self.dropout(m)
        
        if label_feat is not None:
            label_feat_pooled = label_feat.view(label_count, term_count, -1)
            if self.head_pooling == 'max':
                label_feat_pooled = label_feat_pooled.max(dim=1)[0]
            else:
                label_feat_pooled = label_feat_pooled.mean(dim=1)
            
            w = self.w_linear(label_feat_pooled)
            b = self.b_linear(label_feat_pooled)
            
            logits = (m * w).sum(dim=2) + b.squeeze(-1)
        else:
            logits = m.sum(dim=-1)
        
        scores = torch.sigmoid(logits)
        
        if return_attentions:
            return scores, alpha, None
        return scores, None


def create_simple_decoder(config):
    """Create decoder based on config."""
    decoder_name = config.get('name', 'SimpleLAAT')
    
    if decoder_name in ['LAAT', 'SimpleLAAT']:
        return SimpleLAAT(config)
    elif decoder_name in ['CoRelation', 'CoRelationV3', 'CoRelationV4', 'SimpleCoRelation']:
        return SimpleCoRelation(config)
    else:
        return SimpleLAAT(config)
