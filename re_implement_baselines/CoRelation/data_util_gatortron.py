"""
Data utilities for CoRelation with Gatortron encoder.
"""

import os
import re
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import Counter

from config_gatortron import *


class GatortronMimicDataset(Dataset):
    """
    MIMIC dataset with Gatortron tokenizer and chunk processing.
    Aligned with newmimic3 project data format.
    """
    
    def __init__(self, version, mode, tokenizer_name=PRETRAIN_MODEL, 
                 truncate_length=MAX_TEXT_LENGTH, label_truncate_length=LABEL_TRUNCATE_LENGTH,
                 term_count=1, data_path=None):
        self.version = version
        self.mode = mode
        self.truncate_length = truncate_length
        self.label_truncate_length = label_truncate_length
        self.term_count = term_count
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        
        if data_path is None:
            data_path = DATA_PATH
        
        data_type = version.replace("-50", "").replace("_10", "_icd10")
        if version == "mimic4":
            data_type = "mimic4_icd9"
        elif version == "mimic4_10":
            data_type = "mimic4_icd10"
        else:
            data_type = "mimic3"
        
        self.data_path = data_path
        
        if mode == "train":
            data_file = os.path.join(data_path, f"{data_type}_train.pkl")
            label_file = os.path.join(data_path, f"{data_type}_train_1hot.npz")
        elif mode == "dev":
            data_file = os.path.join(data_path, f"{data_type}_val.pkl")
            label_file = os.path.join(data_path, f"{data_type}_val_1hot.npz")
        else:
            data_file = os.path.join(data_path, f"{data_type}_test.pkl")
            label_file = os.path.join(data_path, f"{data_type}_test_1hot.npz")
        
        print(f"Loading data from {data_file}")
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        
        if data_type == "mimic3":
            self.texts = data["TEXT"].tolist()
        else:
            self.texts = data["text"].tolist()
        
        self.labels = np.load(label_file)['arr_0']
        self.code_count = self.labels.shape[1]
        
        print(f"Loaded {len(self.texts)} records, {self.code_count} labels")
        
        self.len = len(self.texts)
        
        label_nums = []
        for i in range(len(self.labels)):
            label_nums.append(self.labels[i].sum())
        self.avg_label_num = np.mean(label_nums)
        
        self.rank_index = self._compute_rank_index()
        
        if mode == "train":
            self._prepare_label_features()
    
    def _compute_rank_index(self):
        """Compute label frequency ranking."""
        label_counts = self.labels.sum(axis=0)
        rank_index = np.argsort(label_counts)[::-1]
        return torch.LongTensor(rank_index)
    
    def _prepare_label_features(self):
        """Prepare label text features for label-aware attention."""
        print("Preparing label features...")
        
        desc_file = os.path.join(self.data_path, f"icd_{self.version.replace('-50', '').replace('_10', '_icd10')}_desc.pkl")
        
        if os.path.exists(desc_file):
            with open(desc_file, "rb") as f:
                label_tokens = pickle.load(f)
            
            self.c_input_ids = label_tokens["input_ids"]
            self.c_attention_mask = label_tokens["attention_mask"]
            self.c_token_type_ids = label_tokens.get("token_type_ids", torch.zeros_like(self.c_input_ids))
        else:
            print(f"Label description file not found: {desc_file}")
            max_label_len = self.label_truncate_length
            self.c_input_ids = torch.zeros(self.code_count, max_label_len, dtype=torch.long)
            self.c_attention_mask = torch.zeros(self.code_count, max_label_len, dtype=torch.long)
            self.c_token_type_ids = torch.zeros(self.code_count, max_label_len, dtype=torch.long)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        input_ids, attention_mask, token_type_ids = self._tokenize_with_chunks(text)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': torch.tensor(label, dtype=torch.float),
        }
    
    def _tokenize_with_chunks(self, text):
        """
        Tokenize text with chunk processing.
        Aligned with newmimic3 project GTCHDataset.
        """
        aux = self.tokenizer(text)
        max_length = self._compute_max_length(aux)
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        
        if max_length > 512:
            input_ids = torch.tensor(encodings['input_ids'])
            attention_mask = torch.tensor(encodings['attention_mask'])
            token_type_ids = torch.tensor(encodings.get('token_type_ids', [0] * len(encodings['input_ids'])))
            
            chunks_input_ids = []
            chunks_attention_mask = []
            chunks_token_type_ids = []
            
            for i in range(0, max_length - (510 - OVERLAP_WINDOW + 1), 510 - OVERLAP_WINDOW):
                chunk_ids = input_ids[i+1:i+511]
                chunk_mask = attention_mask[i+1:i+511]
                chunk_types = token_type_ids[i+1:i+511]
                
                last_token = chunk_ids[-1].item()
                has_content = (last_token != 0) and (last_token != 102)
                
                cls_token = torch.tensor([101])
                sep_token = torch.tensor([102]) if has_content else torch.tensor([0])
                
                chunk_ids = torch.cat([cls_token, chunk_ids, sep_token])
                chunk_mask = torch.cat([torch.tensor([1]), chunk_mask, torch.tensor([1 if has_content else 0])])
                chunk_types = torch.cat([torch.tensor([0]), chunk_types, torch.tensor([0])])
                
                chunks_input_ids.append(chunk_ids)
                chunks_attention_mask.append(chunk_mask)
                chunks_token_type_ids.append(chunk_types)
            
            input_ids = torch.stack(chunks_input_ids)
            attention_mask = torch.stack(chunks_attention_mask)
            token_type_ids = torch.stack(chunks_token_type_ids)
        else:
            input_ids = torch.tensor(encodings['input_ids']).unsqueeze(0)
            attention_mask = torch.tensor(encodings['attention_mask']).unsqueeze(0)
            token_type_ids = torch.tensor(
                encodings.get('token_type_ids', [0] * len(encodings['input_ids']))
            ).unsqueeze(0)
        
        return input_ids.long(), attention_mask.long(), token_type_ids.long()
    
    def _compute_max_length(self, encodings):
        """Compute max length for padding, aligned with newmimic3."""
        lengths = [MIN_TEXT_LENGTH] + list(range(1021, MAX_TEXT_LENGTH, 510))
        num_tokens = len(encodings.input_ids)
        
        if num_tokens <= min(lengths):
            max_length = min(lengths)
        elif num_tokens > max(lengths):
            max_length = max(lengths)
        else:
            max_length = num_tokens
            for n in lengths:
                if max_length <= n:
                    max_length = n
                    break
        return max_length


def gatortron_collate_fn(batch):
    """
    Collate function for batching with variable chunk sizes.
    """
    max_chunks = max(item['input_ids'].size(0) for item in batch)
    chunk_size = batch[0]['input_ids'].size(1)
    batch_size = len(batch)
    
    input_ids = torch.zeros(batch_size, max_chunks, chunk_size, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_chunks, chunk_size, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, max_chunks, chunk_size, dtype=torch.long)
    labels = torch.stack([item['labels'] for item in batch])
    
    for i, item in enumerate(batch):
        num_chunks = item['input_ids'].size(0)
        input_ids[i, :num_chunks] = item['input_ids']
        attention_mask[i, :num_chunks] = item['attention_mask']
        token_type_ids[i, :num_chunks] = item['token_type_ids']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels,
    }


def single_sample_collate_fn(batch):
    """
    Collate function for single sample batch (batch_size=1).
    Returns tensors without batch dimension for easier processing.
    """
    assert len(batch) == 1, "This collate function is for batch_size=1"
    item = batch[0]
    return (
        item['input_ids'],
        item['attention_mask'],
        item['token_type_ids'],
        item['labels'],
    )
