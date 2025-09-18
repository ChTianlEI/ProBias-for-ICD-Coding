import torch 
from torch import nn
from typing import Union, Dict, List, Optional, Tuple, Any
from config import *
from transformers import Trainer
from model_support.eval_metrics import *
from sklearn.metrics import f1_score, recall_score,precision_score
import pickle
import torch.nn.functional as F


class ProBiasTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.pop('labels')
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)

        outputs = model(**inputs)
        logits = outputs
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logits, labels)
        
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
            (loss, (outputs,labels)) =  self.compute_loss(model = model,inputs = inputs,return_outputs = True)
          loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, outputs, labels)

def precision_at_k(y_true,y_pred, k: int=3) -> float:
    head_k_args = np.argsort(np.sum(y_true,axis = 0))[-k:]
    sliced_y_true = np.squeeze(y_true[:,[head_k_args]],axis = 1)
    sliced_y_pred = np.squeeze(y_pred[:,[head_k_args]],axis = 1)
    return precision_score(sliced_y_true,sliced_y_pred, average="micro")

def recall_at_k(y_true,y_pred, k: int=3) -> float:
    head_k_args = np.argsort(np.sum(y_true,axis = 0))[-k:]
    sliced_y_true = np.squeeze(y_true[:,[head_k_args]], axis = 1)
    sliced_y_pred = np.squeeze(y_pred[:,[head_k_args]], axis = 1)
    return recall_score(sliced_y_true,sliced_y_pred, average="micro")

def compute_metrics(pred):
    if os.path.isfile(SAVE_METRICS_PATH + '/metrics_result.pkl'):
        with open(SAVE_METRICS_PATH + '/metrics_result.pkl', 'rb') as f:
            metrics_list = pickle.load(f)
    else:
        metrics_list = []
    labels = pred.label_ids
    logits = pred.predictions

    preds = np.round(1/(1 + np.exp(-logits)))

    #Loss
    loss_list = []
    for logit, label in zip(logits, labels):
      class_loss = torch.nn.BCEWithLogitsLoss(reduction = "mean")(torch.Tensor(logit),torch.Tensor(label))
      loss_list.append(class_loss.item())
    loss = np.mean(loss_list)

    # Metrics
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)

    precision_at_5 = precision_at_k(y_true=labels,y_pred=preds,k=5)
    precision_at_8 = precision_at_k(y_true=labels,y_pred=preds,k=8)
    precision_at_15 = precision_at_k(y_true=labels,y_pred=preds,k=15)

    recall_at_5 = recall_at_k(y_true=labels,y_pred=preds,k=5)
    recall_at_8 = recall_at_k(y_true=labels,y_pred=preds,k=8)
    recall_at_15 = recall_at_k(y_true=labels,y_pred=preds,k=15)
    
    metrics = 'loss: ' + str(loss) + ', f1_macro: ' + str(f1_macro) +  ', f1_micro: ' + str(f1_micro) + ', precision_at_5: ' + str(precision_at_5) +  ', precision_at_8: ' + str(precision_at_8) + ', precision_at_15: ' + str(precision_at_15) + ', recall_at_5: ' + str(recall_at_5) +  ', recall_at_8: ' + str(recall_at_8) + ', recall_at_15: ' + str(recall_at_15)
 
    metrics = metrics + '\n'
    metrics_list.append(metrics)

    with open(SAVE_METRICS_PATH + '/metrics_result.pkl', 'wb') as f:
        pickle.dump(metrics_list, f)
    
    return {
        'loss':loss,
        'f1_macro': f1_macro, 
        'f1_micro': f1_micro,
        'precision_at_5': precision_at_5,
        'precision_at_8': precision_at_8,
        'precision_at_15': precision_at_15,
        'recall_at_5': recall_at_5,
        'recall_at_8': recall_at_8,
        'recall_at_15': recall_at_15
    }
    