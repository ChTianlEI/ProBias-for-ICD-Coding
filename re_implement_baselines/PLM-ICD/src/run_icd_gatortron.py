import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import json
import random
import pickle
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer
)
from torch import nn
from typing import Union, Dict, List, Optional, Tuple, Any

from config import *
from modeling_gatortron import GatortronChunkModel
from evaluation import all_metrics


class ChunkDataset(torch.utils.data.Dataset):
    """
    Dataset with chunk processing aligned with newmimic3 project.
    """
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        aux = self.tokenizer(self.texts[idx])
        max_length = self._compute_max_length(aux)
        encodings = self.tokenizer(
            self.texts[idx], 
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )
        
        if max_length > 512:
            item = {
                k: torch.stack([
                    torch.tensor(v)[i+1:i+511] 
                    for i in range(0, max_length-(510-OVERLAP_WINDOW + 1), 510-OVERLAP_WINDOW)
                ], dim=0) 
                for k, v in encodings.items()
            }
            
            last = item["input_ids"][:, -1]
            sep_tokens = torch.ones((item["input_ids"].shape[0], 1)) * 102
            mask_sep = ((last != 0) * (last != 102)).unsqueeze(1)
            sep_tokens = sep_tokens * mask_sep
            cls_tokens = torch.ones((item["input_ids"].shape[0], 1)) * 101
            
            item["input_ids"] = torch.cat((cls_tokens, item["input_ids"], sep_tokens), dim=1)
            item["token_type_ids"] = torch.cat(
                (torch.zeros(item["token_type_ids"].shape[0], 1), 
                 item["token_type_ids"], 
                 torch.zeros(item["token_type_ids"].shape[0], 1) * mask_sep), 
                dim=1
            )
            item["attention_mask"] = torch.cat(
                (torch.ones(item["attention_mask"].shape[0], 1), 
                 item["attention_mask"], 
                 torch.ones(item["attention_mask"].shape[0], 1) * mask_sep), 
                dim=1
            )
        else:
            item = {k: torch.tensor(v).unsqueeze(0) for k, v in encodings.items()}
        
        item["input_ids"] = item["input_ids"].type(torch.long)
        item["token_type_ids"] = item["token_type_ids"].type(torch.long)
        item["attention_mask"] = item["attention_mask"].type(torch.long)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    def _compute_max_length(self, encodings):
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


class PLMICDTrainer(Trainer):
    """
    Custom trainer for PLM-ICD with chunk processing.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)

        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logits, labels.float().unsqueeze(0))
        
        return (loss, (logits, labels)) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                (loss, (outputs, labels)) = self.compute_loss(model=model, inputs=inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, outputs, labels)


def compute_metrics(pred):
    """Compute evaluation metrics."""
    from sklearn.metrics import f1_score, recall_score, precision_score
    
    labels = pred.label_ids
    logits = pred.predictions

    preds = np.round(1 / (1 + np.exp(-logits)))

    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
    }


def main():
    # Set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
    
    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAVE_METRICS_PATH, exist_ok=True)
    os.makedirs(SAVE_PREDICTION_PATH, exist_ok=True)
    
    # Load data
    print(f"Load {DATA_TYPE} dataset")
    print("Load text...")
    
    train_data_file = DATA_PATH + "/{}_train.pkl".format(DATA_TYPE)
    val_data_file = DATA_PATH + "/{}_val.pkl".format(DATA_TYPE)
    test_data_file = DATA_PATH + "/{}_test.pkl".format(DATA_TYPE)
    
    if DATA_TYPE in ["mimic3"]:
        with open(train_data_file, "rb") as file:
            train_data = pickle.load(file)
            train_texts = train_data["TEXT"].tolist()
        with open(val_data_file, "rb") as file:
            val_data = pickle.load(file)
            val_texts = val_data["TEXT"].tolist()
        with open(test_data_file, "rb") as file:
            test_data = pickle.load(file)
            test_texts = test_data["TEXT"].tolist()
    else:
        with open(train_data_file, "rb") as file:
            train_data = pickle.load(file)
            train_texts = train_data["text"].tolist()
        with open(val_data_file, "rb") as file:
            val_data = pickle.load(file)
            val_texts = val_data["text"].tolist()
        with open(test_data_file, "rb") as file:
            test_data = pickle.load(file)
            test_texts = test_data["text"].tolist()
    
    print("Load 1hot labels...")
    train_1hot_file = DATA_PATH + '/{}_train_1hot.npz'.format(DATA_TYPE)
    val_1hot_file = DATA_PATH + '/{}_val_1hot.npz'.format(DATA_TYPE)
    test_1hot_file = DATA_PATH + '/{}_test_1hot.npz'.format(DATA_TYPE)
    train_1hot = np.load(train_1hot_file)['arr_0']
    val_1hot = np.load(val_1hot_file)['arr_0']
    test_1hot = np.load(test_1hot_file)['arr_0']
    
    # Model config
    num_labels = len(train_1hot[0])
    print(f"Number of labels: {num_labels}")
    
    print("Load Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL, do_lower_case=False)
    
    print("Load Model...")
    config = AutoConfig.from_pretrained(
        PRETRAIN_MODEL, 
        num_labels=num_labels, 
        problem_type="multi_label_classification"
    )
    config.model_name_or_path = PRETRAIN_MODEL
    config.model_mode = MODEL_MODE
    config.chunk_size = CHUNK_SIZE
    
    model = GatortronChunkModel(config)
    
    if START_MODEL_FROM_CHECKPOINT:
        print(f"Loading checkpoint from {START_MODEL_FROM_CHECKPOINT}")
        model = model.from_pretrained(START_MODEL_FROM_CHECKPOINT, config=config)
    
    # Create datasets
    train_dataset = ChunkDataset(train_texts, train_1hot, tokenizer=tokenizer)
    val_dataset = ChunkDataset(val_texts, val_1hot, tokenizer=tokenizer)
    test_dataset = ChunkDataset(test_texts, test_1hot, tokenizer=tokenizer)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=GROUP_BY_LENGTH,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_strategy=LOGGING_STRATEGY,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy=EVALUATION_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        dataloader_drop_last=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        greater_is_better=GREATER_IS_BETTER,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        optim=OPTIM,
        bf16=True,
    )
    
    trainer = PLMICDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    
    if MODE == "train":
        print("======= TRAINING ======")
        print("... Starting to Evaluate model (val set) ...")
        trainer.evaluate(eval_dataset=val_dataset)
        print("... Training model ...")
        trainer.train()
        print("... Evaluate model (test set) ...")
        trainer.evaluate(eval_dataset=test_dataset)
    else:
        print("======= TESTING ======")
        for dataset_name, dataset, y_gt in zip(
            ["val", "test"], 
            [val_dataset, test_dataset], 
            [val_1hot, test_1hot]
        ):
            print(f"Predict {dataset_name} set ...")
            probabilities = np.zeros((len(dataset), num_labels))
            y_pred = np.zeros((len(dataset), num_labels))

            model.eval()
            with tqdm(total=len(dataset)) as pbar:
                for z, item in enumerate(dataset):
                    with torch.no_grad():
                        item = {k: v.to(DEVICE) for k, v in item.items() if k != 'labels'}
                        outputs = model(**item)
                        logits = outputs.logits.cpu().detach().numpy().squeeze()
                    
                    probabilities[z] = 1 / (1 + np.exp(-logits))
                    y_pred[z] = np.round(1 / (1 + np.exp(-logits)))
                    pbar.update(1)

            metrics = all_metrics(y_pred, y_gt, k=[5, 8, 15], yhat_raw=probabilities)
            print(f"{dataset_name} metrics: {metrics}")
            
            np.save(SAVE_PREDICTION_PATH + f'/y_{dataset_name}_prob.npy', probabilities)
            np.save(SAVE_PREDICTION_PATH + f'/y_{dataset_name}_pred.npy', y_pred)
            np.save(SAVE_PREDICTION_PATH + f'/y_{dataset_name}_true.npy', y_gt)

            with open(SAVE_PREDICTION_PATH + f"/{dataset_name}-metrics.txt", "w") as file:
                file.write(str(metrics))


if __name__ == '__main__':
    main()
