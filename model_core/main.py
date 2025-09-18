import os
import numpy as np
import torch
import json
import random
from tqdm import tqdm
import pickle

from transformers import ( 
    AutoConfig, 
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from config import *
from models.model import Model
from model_support.dataset import ProBiasDataset
from model_support.trainer import ProBiasTrainer, compute_metrics
from model_support.eval_metrics import all_metrics

def main():

    # ========= Set seed =============
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
    
    # ========= Create some files =============

    if not os.path.exists("../save/{}".format(DATA_TYPE)):
        os.makedirs("../save/{}".format(DATA_TYPE))
    if not os.path.exists("../save/{}/{}".format(DATA_TYPE,FILE_NAME)):
        os.makedirs("../save/{}/{}".format(DATA_TYPE,FILE_NAME))
        os.makedirs(SAVE_METRICS_PATH)    
        os.makedirs(SAVE_PREDICTION_PATH)
        os.makedirs(OUTPUT_DIR)

    # ========== Load data =======
    print("Load {} dataset".format(DATA_TYPE))
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

    # ======== CONFIG ========
    num_labels = len(train_1hot[0])

    print("Load Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL, do_lower_case=False)

    print("Load Model...")

    config = AutoConfig.from_pretrained(PRETRAIN_MODEL, num_labels=num_labels, problem_type="multi_label_classification")
    config.model_name = PRETRAIN_MODEL
    config.attention_hidden_size = HIDDEN_SIZE
    config.transform_size = TRANSFORM_SIZE
    config.n_heads = NUM_ATT_HEAD

    config.code_token_file = "../model_data/{}/icd_{}_{}.pkl".format(DATA_TYPE, DATA_TYPE,CODE_TYPE)

    config.adj_matrix_file = "../model_data/{}/{}_adj_matrix.pkl".format(DATA_TYPE, DATA_TYPE)
    config.c_indices_file = "../model_data/{}/{}_c_indices.pkl".format(DATA_TYPE, DATA_TYPE)
    
    print(f"Using device: {DEVICE}")
    config.ground_ind_tail_file = "../model_data/{}/{}_ground_ind_tail.pkl".format(DATA_TYPE, DATA_TYPE)
    config.ground_ind_head_file = "../model_data/{}/{}_ground_ind_head.pkl".format(DATA_TYPE, DATA_TYPE)
    config.graph_att_hidd = GRAPH_ATT
    config.graph_ffn_hidd = GRAPH_FFN

    # ======= DATA TYPE SETTING =======
    if DATA_TYPE == "mimic4_icd9":
        config.head_num = 2345
        config.tail_num = 8986
    if DATA_TYPE == "mimic4_icd10":
        config.head_num = 7613
        config.tail_num = 18483
    if DATA_TYPE == "mimic3":
        config.head_num = 3537
        config.tail_num = 5384

    # ======= START MODEL FROM CHECKPOINT or NOT =======
    model = Model(config=config)
    if START_MODEL_FROM_CHECKPOINT != "":
        model = model.from_pretrained(START_MODEL_FROM_CHECKPOINT, config = config)

    # ======= DATASETS SETTING =======
    train_dataset = ProBiasDataset(train_texts, train_1hot,tokenizer = tokenizer)
    val_dataset = ProBiasDataset(val_texts, val_1hot,tokenizer = tokenizer)
    test_dataset = ProBiasDataset(test_texts, test_1hot,tokenizer = tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

    # ======= Mixed Precision Training with BF16 ======

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, 
        group_by_length=GROUP_BY_LENGTH,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_strategy = LOGGING_STRATEGY,
        num_train_epochs=EPOCHS,                               
        per_device_train_batch_size=1,  
        per_device_eval_batch_size=1,                    
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy=EVALUATION_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        dataloader_drop_last = True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        greater_is_better =GREATER_IS_BETTER,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        optim=OPTIM,
        bf16=True
    )


    trainer = ProBiasTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )


    try:
        json.dumps(model.config.to_dict())
        print("config is serializable!")
    except TypeError as e:
        print("still has problem:", e)

    if MODE == "train":
        print("======= LETS TRAIN ======")
        print("... Starting to Evaluate model (val set) ...")
        trainer.evaluate(eval_dataset = val_dataset)
        print("... Training model ...")
        trainer.train()
        print("... Evaluate model (test set) ...")
        trainer.evaluate(eval_dataset = test_dataset)
    else:
        print("======= LETS TEST ======")
        for dataset,texts,y_gt in zip(["val","test"],[val_dataset,test_dataset],[val_1hot,test_1hot]):
            print("Predict {} set ...".format(dataset))
            probabilities = np.zeros((len(texts),num_labels))

            y_pred = np.zeros((len(texts),num_labels))

            with tqdm(total=len(texts)) as pbar:
                for z, item in enumerate(texts):
                    model.eval()
                    with torch.no_grad():
                        item = {k: v.to(DEVICE) for k, v in item.items()}
                        logits = model(**item)[0].cpu().detach().numpy()
                        
                    probabilities[z] = 1/(1 + np.exp(-logits))
                    
                    y_pred[z] = np.round(1/(1 + np.exp(-logits)))

                    pbar.update(1)

            metrics = all_metrics(y_pred, y_gt, k=[5,8,15], yhat_raw=probabilities)
            np.save(SAVE_PREDICTION_PATH + '/y_{}_prob.npy'.format(dataset), probabilities)

            np.save(SAVE_PREDICTION_PATH + '/y_{}_pred.npy'.format(dataset), y_pred)

            np.save(SAVE_PREDICTION_PATH + '/y_{}_true.npy'.format(dataset), y_gt)

            with open(SAVE_PREDICTION_PATH + "/{}-metrics.txt".format(dataset), "w") as file:
                file.write(str(metrics))
            

if __name__ == '__main__':
    main()
