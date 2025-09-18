import torch
from config import *

class ProBiasDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        aux = self.tokenizer(self.texts[idx])
        max_length = compute_max_length(aux)
        encodings = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=max_length)
        if max_length > 512:
            item = {k: torch.stack([torch.tensor(v)[i+1:i+511] for i in range(0,max_length-(510-OVERLAP_WINDOW + 1),510-OVERLAP_WINDOW)],dim=0) for k, v in encodings.items()}
            last = item["input_ids"][:,-1]
            sep_tokens = torch.ones((item["input_ids"].shape[0],1))*102
            mask_sep = ((last != 0)*(last != 102)).unsqueeze(1)
            sep_tokens = sep_tokens*mask_sep
            cls_tokens = torch.ones((item["input_ids"].shape[0],1))*101
            item["input_ids"] = torch.cat((cls_tokens,item["input_ids"],sep_tokens), dim = 1)
            item["token_type_ids"] = torch.cat((torch.zeros(item["token_type_ids"].shape[0],1),item["token_type_ids"],torch.zeros(item["token_type_ids"].shape[0],1)*mask_sep), dim = 1)
            item["attention_mask"] = torch.cat((torch.ones(item["attention_mask"].shape[0],1),item["attention_mask"],torch.ones(item["attention_mask"].shape[0],1)*mask_sep), dim = 1)
        else:
            item = {k: torch.tensor(v).unsqueeze(0) for k, v in encodings.items()}
        item["input_ids"] = item["input_ids"].type(torch.long)
        item["token_type_ids"] = item["token_type_ids"].type(torch.long)
        item["attention_mask"] = item["attention_mask"].type(torch.long)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def compute_max_length(encodings):
    lengths = [MIN_TEXT_LENGTH] + list(range(1021,MAX_TEXT_LENGTH,510))
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