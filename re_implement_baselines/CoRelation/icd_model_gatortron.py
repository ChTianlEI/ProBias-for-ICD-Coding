"""
ICD Model with Gatortron encoder for CoRelation.
"""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config_gatortron import *
from models.decoder_simple import create_simple_decoder
from models.losses import loss_fn


def compute_kl_loss(p, q, label_avg_num=None, require_activation=False):
    """KL divergence loss for R-Drop regularization."""
    p = p.contiguous()
    q = q.contiguous()
    
    if require_activation:
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    else:
        p_loss = F.kl_div(torch.log(p.clamp(min=1e-9)), q, reduction='none')
        q_loss = F.kl_div(torch.log(q.clamp(min=1e-9)), p, reduction='none')
    
    if label_avg_num is not None:
        p_loss = (p_loss.sum(dim=1) / label_avg_num).mean()
        q_loss = (q_loss.sum(dim=1) / label_avg_num).mean()
    else:
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()
    
    return (p_loss + q_loss) / 2


class GatortronEncoder(nn.Module):
    """Gatortron-based text encoder with chunk processing."""
    
    def __init__(self, model_name, output_dim, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        if self.bert_hidden_size != output_dim:
            self.proj = nn.Linear(self.bert_hidden_size, output_dim)
        else:
            self.proj = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args:
            input_ids: (num_chunks, chunk_size) for single sample
            attention_mask: (num_chunks, chunk_size)
            token_type_ids: (num_chunks, chunk_size)
        
        Returns:
            hidden: (num_chunks, chunk_size, hidden_dim)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        
        hidden = outputs.last_hidden_state
        
        if self.proj is not None:
            hidden = self.proj(hidden)
        
        hidden = self.dropout(hidden)
        
        return hidden


class GatortronLabelEncoder(nn.Module):
    """Encoder for ICD code descriptions using Gatortron."""
    
    def __init__(self, model_name, output_dim, pooling='mean'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.pooling = pooling
        
        if self.bert_hidden_size != output_dim:
            self.proj = nn.Linear(self.bert_hidden_size, output_dim)
        else:
            self.proj = None
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args:
            input_ids: (num_labels * term_count, seq_len)
            attention_mask: (num_labels * term_count, seq_len)
        
        Returns:
            label_repr: (num_labels * term_count, hidden_dim)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        
        if self.pooling == 'cls':
            hidden = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'mean':
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            hidden = outputs.last_hidden_state[:, 0, :]
        
        if self.proj is not None:
            hidden = self.proj(hidden)
        
        return hidden


class IcdModelGatortron(nn.Module):
    """
    ICD coding model with Gatortron encoder.
    Aligned with CoRelation architecture but uses Gatortron instead of word2vec.
    """
    
    def __init__(self, args, num_labels):
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        
        self.encoder = GatortronEncoder(
            model_name=args.model_name,
            output_dim=args.rnn_dim,
            dropout=args.dropout
        )
        
        decoder_config = {
            'name': args.decoder,
            'input_dim': args.rnn_dim,
            'attention_dim': args.attention_dim,
            'attention_head': args.attention_head,
            'attention_head_dim': args.attention_head_dim,
            'num_labels': num_labels,
            'text_pooling': args.text_pooling,
            'head_pooling': args.head_pooling,
        }
        self.decoder = create_simple_decoder(decoder_config)
        
        self.loss_config = {
            'name': args.loss_name,
            'code_loss_weight': 1.0,
            'main_code_loss_weight': args.main_code_loss_weight,
            'rdrop_alpha': args.rdrop_alpha,
            'alpha_weight': args.alpha_weight,
        }
        
        self.topk_num = args.topk_num
        self.kl_loss_fn = compute_kl_loss
        
        self.label_feats = None
        self.mlabel_feats = None
        self.c_input_ids = None
        self.c_attention_mask = None
        self.avg_label_num = None
        self.rank_index = None
    
    def calculate_text_hidden(self, input_ids, attention_mask, token_type_ids=None):
        """Encode text using Gatortron."""
        return self.encoder(input_ids, attention_mask, token_type_ids)
    
    def calculate_label_hidden(self):
        """Pre-compute label representations."""
        if self.c_input_ids is None:
            return
        
        with torch.no_grad():
            batch_size = 64
            all_feats = []
            
            for i in range(0, self.c_input_ids.size(0), batch_size):
                batch_ids = self.c_input_ids[i:i+batch_size]
                batch_mask = self.c_attention_mask[i:i+batch_size]
                
                outputs = self.encoder.bert(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    return_dict=True,
                )
                
                hidden = outputs.last_hidden_state
                mask = batch_mask.unsqueeze(-1).float()
                feats = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                
                if self.encoder.proj is not None:
                    feats = self.encoder.proj(feats)
                
                all_feats.append(feats)
            
            self.label_feats = torch.cat(all_feats, dim=0)
    
    def calculate_label_hidden_m(self):
        """For graph-based label encoding (placeholder)."""
        self.mlabel_feats = self.label_feats
    
    def forward(self, batch, rdrop=False, indices=None):
        """
        Forward pass.
        
        Args:
            batch: tuple of (input_ids, attention_mask, token_type_ids, labels)
            rdrop: whether to use R-Drop regularization
        """
        if rdrop:
            return self.forward_rdrop(batch, indices)
        else:
            return self.forward_normal(batch)
    
    def forward_normal(self, batch):
        """Normal forward pass without R-Drop."""
        input_ids, attention_mask, token_type_ids, labels = batch[:4]
        
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.squeeze(0)
        
        hidden = self.calculate_text_hidden(input_ids, attention_mask, token_type_ids)
        
        hidden_flat = hidden.view(-1, hidden.size(-1))
        mask_flat = attention_mask.view(-1)
        
        if self.label_feats is not None:
            c_logits, c_alphas = self.decoder(
                hidden_flat.unsqueeze(0),
                mask_flat.unsqueeze(0),
                self.label_feats,
                self.args.term_count,
                mlabel_feat=self.mlabel_feats
            )
        else:
            c_logits, c_alphas = self.decoder(
                hidden_flat.unsqueeze(0),
                mask_flat.unsqueeze(0),
                None,
                1
            )
        
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        
        c_loss = loss_fn(c_logits, labels, self.loss_config)
        loss = c_loss * self.loss_config['code_loss_weight']
        
        if c_alphas is not None and self.loss_config['alpha_weight'] != 0.0:
            a_loss = c_alphas.mean()
            loss += self.loss_config['alpha_weight'] * a_loss
        else:
            a_loss = torch.tensor(0.0)
        
        return {
            'loss': loss,
            'c_loss': c_loss * self.loss_config['code_loss_weight'],
            'alpha_loss': self.loss_config['alpha_weight'] * a_loss,
            'kl_loss': torch.tensor(0.0),
            'indices_next': None,
        }
    
    def forward_rdrop(self, batch, indices=None):
        """Forward pass with R-Drop regularization."""
        input_ids, attention_mask, token_type_ids, labels = batch[:4]
        
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.squeeze(0)
        
        hidden0 = self.calculate_text_hidden(input_ids, attention_mask, token_type_ids)
        hidden1 = self.calculate_text_hidden(input_ids, attention_mask, token_type_ids)
        
        hidden0_flat = hidden0.view(-1, hidden0.size(-1))
        hidden1_flat = hidden1.view(-1, hidden1.size(-1))
        mask_flat = attention_mask.view(-1)
        
        term_count = self.args.term_count
        
        if self.label_feats is not None:
            c_logits0, c_alphas0 = self.decoder(
                hidden0_flat.unsqueeze(0), mask_flat.unsqueeze(0),
                self.label_feats, term_count, mlabel_feat=self.mlabel_feats
            )
            c_logits1, c_alphas1 = self.decoder(
                hidden1_flat.unsqueeze(0), mask_flat.unsqueeze(0),
                self.label_feats, term_count, mlabel_feat=self.mlabel_feats
            )
        else:
            c_logits0, c_alphas0 = self.decoder(
                hidden0_flat.unsqueeze(0), mask_flat.unsqueeze(0), None, 1
            )
            c_logits1, c_alphas1 = self.decoder(
                hidden1_flat.unsqueeze(0), mask_flat.unsqueeze(0), None, 1
            )
        
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        
        c_loss = (loss_fn(c_logits0, labels, self.loss_config) +
                  loss_fn(c_logits1, labels, self.loss_config)) * 0.5
        
        kl_loss = self.kl_loss_fn(
            torch.sigmoid(c_logits0),
            torch.sigmoid(c_logits1),
            self.avg_label_num
        )
        
        loss = self.loss_config['rdrop_alpha'] * kl_loss + c_loss * self.loss_config['code_loss_weight']
        
        if c_alphas0 is not None and c_alphas1 is not None and self.loss_config['alpha_weight'] != 0.0:
            a_loss = (c_alphas0.mean() + c_alphas1.mean()) * 0.5
            loss += self.loss_config['alpha_weight'] * a_loss
        else:
            a_loss = torch.tensor(0.0)
        
        return {
            'loss': loss,
            'c_loss': c_loss * self.loss_config['code_loss_weight'],
            'kl_loss': self.loss_config['rdrop_alpha'] * kl_loss,
            'alpha_loss': self.loss_config['alpha_weight'] * a_loss,
            'indices_next': None,
        }
    
    def predict(self, batch, threshold=0.5):
        """Prediction for evaluation."""
        input_ids, attention_mask, token_type_ids, labels = batch[:4]
        
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.squeeze(0)
        
        hidden = self.calculate_text_hidden(input_ids, attention_mask, token_type_ids)
        hidden_flat = hidden.view(-1, hidden.size(-1))
        mask_flat = attention_mask.view(-1)
        
        if self.label_feats is not None:
            yhat_raw, _ = self.decoder(
                hidden_flat.unsqueeze(0), mask_flat.unsqueeze(0),
                self.label_feats, self.args.term_count,
                mlabel_feat=self.mlabel_feats
            )
        else:
            yhat_raw, _ = self.decoder(
                hidden_flat.unsqueeze(0), mask_flat.unsqueeze(0), None, 1
            )
        
        yhat_raw = torch.sigmoid(yhat_raw)
        yhat = (yhat_raw >= threshold).int()
        
        return {
            'yhat_raw': yhat_raw.cpu().detach().numpy(),
            'yhat': yhat.cpu().detach().numpy(),
            'y': labels.cpu().detach().numpy() if labels.dim() > 1 else labels.unsqueeze(0).cpu().detach().numpy(),
        }
    
    def configure_optimizers(self, train_dataloader=None):
        """Configure optimizer and scheduler."""
        from transformers import get_linear_schedule_with_warmup, AdamW
        
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate
            },
        ]
        
        optimizer = AdamW(params, eps=1e-8)
        
        if train_dataloader is not None:
            total_steps = len(train_dataloader) * self.args.train_epoch
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_steps * self.args.warmup_ratio),
                num_training_steps=total_steps,
            )
            return [optimizer], [scheduler]
        
        return [optimizer], [None]
