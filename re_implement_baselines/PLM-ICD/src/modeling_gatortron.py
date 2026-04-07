# coding=utf-8
import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss

from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel


class GatortronForMultilabelClassification(PreTrainedModel):
    """
    Gatortron model with LAAT attention for multi-label classification.
    Supports chunk-based processing for long documents.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model_mode = getattr(config, 'model_mode', 'laat')
        
        self.bert = AutoModel.from_pretrained(
            config.model_name_or_path,
            config=config,
            add_pooling_layer=False
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if "cls" in self.model_mode:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        elif "laat" in self.model_mode:
            self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
            self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")
        
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Args:
            input_ids: (batch_size, num_chunks, chunk_size)
            attention_mask: (batch_size, num_chunks, chunk_size)
            token_type_ids: (batch_size, num_chunks, chunk_size)
            labels: (batch_size, num_labels)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, num_chunks, chunk_size = input_ids.size()
        
        outputs = self.bert(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size) if attention_mask is not None else None,
            token_type_ids=token_type_ids.view(-1, chunk_size) if token_type_ids is not None else None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if "cls" in self.model_mode:
            pooled_output = outputs.last_hidden_state[:, 0, :].view(batch_size, num_chunks, -1)
            if self.model_mode == "cls-sum":
                pooled_output = pooled_output.sum(dim=1)
            elif self.model_mode == "cls-max":
                pooled_output = pooled_output.max(dim=1).values
            else:
                raise ValueError(f"model_mode {self.model_mode} not recognized")
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
        elif "laat" in self.model_mode:
            if self.model_mode == "laat":
                hidden_output = outputs.last_hidden_state.view(batch_size, num_chunks * chunk_size, -1)
            elif self.model_mode == "laat-split":
                hidden_output = outputs.last_hidden_state.view(batch_size * num_chunks, chunk_size, -1)
            
            weights = torch.tanh(self.first_linear(hidden_output))
            att_weights = self.second_linear(weights)
            att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
            weighted_output = att_weights @ hidden_output
            logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)
            
            if self.model_mode == "laat-split":
                logits = logits.view(batch_size, num_chunks, -1).max(dim=1).values
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")
        
        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GatortronChunkModel(PreTrainedModel):
    """
    Gatortron model with chunk processing aligned with newmimic3 project.
    Uses MaxPool aggregation across chunks like the main GTCH model.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.chunk_size = getattr(config, 'chunk_size', 512)
        
        self.bert = AutoModel.from_pretrained(
            config.model_name_or_path,
            config=config,
            add_pooling_layer=False
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
        
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        return_dict=None,
    ):
        """
        Args:
            input_ids: (num_chunks, chunk_size) - single sample with multiple chunks
            attention_mask: (num_chunks, chunk_size)
            token_type_ids: (num_chunks, chunk_size)
            labels: (num_labels,)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0) if attention_mask is not None else None
            token_type_ids = token_type_ids.squeeze(0) if token_type_ids is not None else None
        
        num_chunks = input_ids.size(0)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        
        hidden_output = outputs.last_hidden_state
        
        weights = torch.tanh(self.first_linear(hidden_output))
        att_weights = self.second_linear(weights)
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ hidden_output
        
        chunk_logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)
        
        logits = chunk_logits.max(dim=0).values.unsqueeze(0)
        
        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            loss = loss_fct(logits, labels.float())
        
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
