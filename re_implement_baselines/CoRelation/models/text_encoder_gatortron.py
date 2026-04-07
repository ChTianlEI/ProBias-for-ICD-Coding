import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class GatortronTextEncoder(nn.Module):
    """
    Text encoder using Gatortron with chunk processing.
    """
    
    def __init__(self, config):
        super(GatortronTextEncoder, self).__init__()
        self.config = config
        self.model_name = config.get('model_name', 'UFNLP/gatortron-base')
        self.hidden_size = config.get('hidden_size', 1024)
        self.output_dim = config.get('output_dim', 512)
        self.dropout_rate = config.get('dropout', 0.1)
        self.freeze_encoder = config.get('freeze_encoder', False)
        
        self.bert = AutoModel.from_pretrained(self.model_name)
        
        if self.freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        if self.bert_hidden_size != self.output_dim:
            self.proj = nn.Linear(self.bert_hidden_size, self.output_dim)
        else:
            self.proj = None
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len) or (batch_size, num_chunks, chunk_size)
            attention_mask: same shape as input_ids
            token_type_ids: same shape as input_ids (optional)
        
        Returns:
            hidden: (batch_size, seq_len, hidden_dim) or aggregated representation
        """
        if input_ids.dim() == 3:
            batch_size, num_chunks, chunk_size = input_ids.size()
            
            input_ids_flat = input_ids.view(-1, chunk_size)
            attention_mask_flat = attention_mask.view(-1, chunk_size) if attention_mask is not None else None
            token_type_ids_flat = token_type_ids.view(-1, chunk_size) if token_type_ids is not None else None
            
            outputs = self.bert(
                input_ids=input_ids_flat,
                attention_mask=attention_mask_flat,
                token_type_ids=token_type_ids_flat,
                return_dict=True,
            )
            
            hidden = outputs.last_hidden_state
            hidden = hidden.view(batch_size, num_chunks * chunk_size, -1)
            
            if attention_mask is not None:
                attention_mask = attention_mask.view(batch_size, -1)
        else:
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


class GatortronChunkEncoder(nn.Module):
    """
    Gatortron encoder with chunk processing for long documents.
    Each chunk is processed separately and then aggregated.
    """
    
    def __init__(self, config):
        super(GatortronChunkEncoder, self).__init__()
        self.config = config
        self.model_name = config.get('model_name', 'UFNLP/gatortron-base')
        self.output_dim = config.get('output_dim', 512)
        self.dropout_rate = config.get('dropout', 0.1)
        self.chunk_size = config.get('chunk_size', 512)
        self.overlap_window = config.get('overlap_window', 255)
        
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        if self.bert_hidden_size != self.output_dim:
            self.proj = nn.Linear(self.bert_hidden_size, self.output_dim)
        else:
            self.proj = None
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, return_chunks=False):
        """
        Process input with chunk-based encoding.
        
        Args:
            input_ids: (num_chunks, chunk_size)
            attention_mask: (num_chunks, chunk_size)
            token_type_ids: (num_chunks, chunk_size)
            return_chunks: if True, return per-chunk representations
        
        Returns:
            hidden: (total_tokens, hidden_dim) or (num_chunks, chunk_size, hidden_dim)
        """
        num_chunks = input_ids.size(0)
        
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
        
        if return_chunks:
            return hidden
        
        hidden = hidden.view(-1, hidden.size(-1))
        
        if attention_mask is not None:
            mask = attention_mask.view(-1).bool()
            hidden = hidden[mask]
        
        return hidden
    
    def get_cls_representation(self, input_ids, attention_mask, token_type_ids=None):
        """
        Get CLS token representation for each chunk and aggregate.
        
        Returns:
            cls_repr: (hidden_dim,) aggregated CLS representation
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        
        if self.proj is not None:
            cls_hidden = self.proj(cls_hidden)
        
        aggregated = cls_hidden.max(dim=0).values
        
        return aggregated


class GatortronLabelEncoder(nn.Module):
    """
    Encoder for ICD code descriptions using Gatortron.
    Used to encode label text for label-aware attention.
    """
    
    def __init__(self, config):
        super(GatortronLabelEncoder, self).__init__()
        self.config = config
        self.model_name = config.get('model_name', 'UFNLP/gatortron-base')
        self.output_dim = config.get('output_dim', 512)
        self.pooling = config.get('pooling', 'cls')
        
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        if self.bert_hidden_size != self.output_dim:
            self.proj = nn.Linear(self.bert_hidden_size, self.output_dim)
        else:
            self.proj = None
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Encode label descriptions.
        
        Args:
            input_ids: (num_labels, seq_len)
            attention_mask: (num_labels, seq_len)
        
        Returns:
            label_repr: (num_labels, hidden_dim)
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
        elif self.pooling == 'max':
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            hidden = hidden.masked_fill(mask == 0, -1e9)
            hidden = hidden.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        if self.proj is not None:
            hidden = self.proj(hidden)
        
        return hidden
