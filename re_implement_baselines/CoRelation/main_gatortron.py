"""
CoRelation with Gatortron encoder.
"""

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import random
import json
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

from config_gatortron import *
from data_util_gatortron import GatortronMimicDataset, single_sample_collate_fn
from icd_model_gatortron import IcdModelGatortron
from evaluation import all_metrics, print_metrics
from find_threshold import find_threshold_micro


def parse_args():
    parser = argparse.ArgumentParser(description="CoRelation with Gatortron")
    
    parser.add_argument("--version", type=str, default=VERSION)
    parser.add_argument("--model_name", type=str, default=PRETRAIN_MODEL)
    parser.add_argument("--rnn_dim", type=int, default=RNN_DIM)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--decoder", type=str, default="CoRelationV4")
    parser.add_argument("--attention_head", type=int, default=ATTENTION_HEAD)
    parser.add_argument("--attention_head_dim", type=int, default=ATTENTION_HEAD_DIM)
    parser.add_argument("--attention_dim", type=int, default=ATTENTION_DIM)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--train_epoch", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--main_code_loss_weight", type=float, default=0.0)
    parser.add_argument("--rdrop_alpha", type=float, default=RDROP_ALPHA)
    parser.add_argument("--term_count", type=int, default=TERM_COUNT)
    parser.add_argument("--head_pooling", type=str, default=HEAD_POOLING)
    parser.add_argument("--text_pooling", type=str, default=TEXT_POOLING)
    parser.add_argument("--alpha_weight", type=float, default=ALPHA_WEIGHT)
    parser.add_argument("--use_graph", action="store_true", default=USE_GRAPH)
    parser.add_argument("--topk_num", type=int, default=TOPK_NUM)
    parser.add_argument("--output_base_dir", type=str, default=OUTPUT_BASE_DIR)
    parser.add_argument("--loss_name", type=str, default="bce")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stop_epoch", type=int, default=5)
    parser.add_argument("--early_stop_metric", type=str, default="f1_micro")
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    
    return parser.parse_args()


def train_one_epoch(model, train_dataloader, optimizer, scheduler, args, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    epoch_c_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_alpha_loss = 0.0
    
    epoch_iterator = tqdm(train_dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(epoch_iterator):
        batch_gpu = tuple([x.to(device) if isinstance(x, torch.Tensor) else x for x in batch])
        
        ori_loss = model(batch_gpu, rdrop=args.rdrop_alpha > 0.0)
        loss = ori_loss['loss']
        
        batch_loss = float(loss.item())
        epoch_loss += batch_loss
        epoch_c_loss += float(ori_loss['c_loss'].item())
        
        if args.rdrop_alpha > 0.0:
            epoch_kl_loss += float(ori_loss['kl_loss'].item())
        
        epoch_alpha_loss += float(ori_loss['alpha_loss'].item()) if ori_loss['alpha_loss'].item() else 0.0
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.zero_grad()
        
        epoch_iterator.set_description(
            f"Loss: {epoch_loss / (batch_idx + 1):.4f}, "
            f"C_Loss: {epoch_c_loss / (batch_idx + 1):.4f}"
        )
    
    return epoch_loss / len(train_dataloader)


def evaluate(model, dataloader, device, threshold=None, args=None):
    """Evaluate model."""
    model.eval()
    
    all_yhat = []
    all_yhat_raw = []
    all_y = []
    
    with torch.no_grad():
        model.calculate_label_hidden()
        if args.use_graph:
            model.calculate_label_hidden_m()
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_gpu = tuple([x.to(device) if isinstance(x, torch.Tensor) else x for x in batch])
            
            result = model.predict(batch_gpu, threshold=threshold if threshold is not None else args.prob_threshold)
            
            all_yhat.append(result['yhat'])
            all_yhat_raw.append(result['yhat_raw'])
            all_y.append(result['y'])
    
    yhat = np.concatenate(all_yhat, axis=0)
    yhat_raw = np.concatenate(all_yhat_raw, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    if threshold is None:
        threshold = find_threshold_micro(yhat_raw, y)
        yhat = np.where(yhat_raw >= threshold, 1, 0)
    
    metrics = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)
    
    return metrics, threshold


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_name = f"gatortron_{args.version}_rdrop{args.rdrop_alpha}_seed{args.seed}"
    output_path = os.path.join(args.output_base_dir, output_name)
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading datasets...")
    train_dataset = GatortronMimicDataset(
        args.version, "train",
        tokenizer_name=args.model_name,
        data_path=args.data_path
    )
    dev_dataset = GatortronMimicDataset(
        args.version, "dev",
        tokenizer_name=args.model_name,
        data_path=args.data_path
    )
    test_dataset = GatortronMimicDataset(
        args.version, "test",
        tokenizer_name=args.model_name,
        data_path=args.data_path
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=1,
        collate_fn=single_sample_collate_fn,
        shuffle=True,
        num_workers=2
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=1,
        collate_fn=single_sample_collate_fn,
        shuffle=False,
        num_workers=2
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=single_sample_collate_fn,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Number of labels: {train_dataset.code_count}")
    
    print("Creating model...")
    model = IcdModelGatortron(args, train_dataset.code_count).to(device)
    
    if hasattr(train_dataset, 'c_input_ids') and train_dataset.c_input_ids is not None:
        model.c_input_ids = train_dataset.c_input_ids.to(device)
        model.c_attention_mask = train_dataset.c_attention_mask.to(device)
    
    model.avg_label_num = train_dataset.avg_label_num
    model.rank_index = train_dataset.rank_index.to(device)
    
    optimizer, scheduler = model.configure_optimizers(train_dataloader)
    optimizer = optimizer[0]
    scheduler = scheduler[0]
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_dev_metric = None
    best_test_metric = None
    best_epoch = 0
    early_stop_count = 0
    
    for epoch in range(1, args.train_epoch + 1):
        print(f"\n===== Epoch {epoch}/{args.train_epoch} =====")
        
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, args, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        dev_metric, threshold = evaluate(model, dev_dataloader, device, args=args)
        print_metrics(dev_metric, f"Dev_Epoch{epoch}")
        
        test_metric, _ = evaluate(model, test_dataloader, device, threshold=threshold, args=args)
        print_metrics(test_metric, f"Test_Epoch{epoch}")
        
        torch.save(model.state_dict(), os.path.join(output_path, f"epoch{epoch}.pth"))
        
        if best_dev_metric is None or dev_metric[args.early_stop_metric] >= best_dev_metric[args.early_stop_metric]:
            best_dev_metric = dev_metric
            best_test_metric = test_metric
            best_epoch = epoch
            early_stop_count = 0
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pth"))
        else:
            early_stop_count += 1
        
        if args.early_stop_epoch > 0 and early_stop_count >= args.early_stop_epoch:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\n===== Best Results (Epoch {best_epoch}) =====")
    print_metrics(best_dev_metric, "Best_Dev")
    print_metrics(best_test_metric, "Best_Test")
    
    with open(os.path.join(output_path, 'best_metrics.json'), 'w') as f:
        json.dump({
            'best_epoch': best_epoch,
            'dev': best_dev_metric,
            'test': best_test_metric
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
