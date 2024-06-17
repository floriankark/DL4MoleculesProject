import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
import pandas as pd
import numpy as np
import wandb
import os
os.environ["WANDB_MODE"]="offline" # HPC clusters are not connected to the internet
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disabling parallelism to avoid deadlocks
from torch.utils.data import Dataset
from datasets import Dataset
import argparse

BATCH_SIZE = 16

train = pd.read_csv('/Users/floriankark/Desktop/project/data/BBB_test.csv', index_col=0, dtype={'Drug': str, 'Y': int})
train['text'] = train['Drug'].astype(str)
train['label'] = train['Y'].apply(lambda x: 1. if x == "Yes" else 0.)
train = train[['text', 'label']]

val = pd.read_csv('/Users/floriankark/Desktop/project/data/BBB_valid.csv', index_col=0, dtype={'Drug': str, 'Y': int})
val['text'] = val['Drug'].astype(str)
val['label'] = val['Y'].apply(lambda x: 1. if x == "Yes" else 0.)
val = val[['text', 'label']]

test = pd.read_csv('/Users/floriankark/Desktop/project/data/BBB_test.csv', index_col=0, dtype={'Drug': str, 'Y': int})
test['text'] = test['Drug'].astype(str)
test['label'] = test['Y'].apply(lambda x: 1. if x == "Yes" else 0.)
test = test[['text', 'label']]

train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)
test_dataset = Dataset.from_pandas(test)

def process_example(examples):
    texts = [t for t in examples['text']]

    text_inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=tokenizer.model_max_length)
    
    examples['input_ids'] = text_inputs['input_ids']
    examples['attention_mask'] = text_inputs['attention_mask']
    return examples

train_dataset = train_dataset.map(process_example, batched=True, batch_size=32,
                                  remove_columns=['text'])

val_dataset = val_dataset.map(process_example, batched=True, batch_size=32,
                              remove_columns=['text'])

test_dataset = test_dataset.map(process_example, batched=True, batch_size=32,
                                remove_columns=['text'])

train_dataset.save_to_disk("train_dataset")
val_dataset.save_to_disk("dev_dataset")
test_dataset.save_to_disk("test_dataset")
    
class ClassificationModel(nn.Module):
    def __init__(self,
                 model,
                 loss_fct=torch.nn.CrossEntropyLoss(),
                 ):
        
        super(ClassificationModel, self).__init__()

        self.model = model

        hidden_size = self.text_model.config.hidden_size
        dropout_prob = self.text_model.config.hidden_dropout_prob

        self.classifier = nn.Sequential(nn.Dropout(dropout_prob),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(dropout_prob),
                                        nn.Linear(hidden_size, 2),
                                        )

        self.loss_fct = loss_fct

    def forward(self, 
                input_ids=None,
                attention_mask=None,
                token_type_ids=None, 
                position_ids=None, 
                head_mask=None,
                labels=None
                ):

        outputs = self.text_model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                #head_mask=head_mask,
                                )

        text_features = outputs.last_hidden_state

        text_representation = text_features[:, 0, :]

        logits = self.classifier(text_representation)
        
        loss = self.loss_fct(logits.view(-1, 2), labels.view(-1).long())

        return SequenceClassifierOutput(loss=loss, 
                                        logits=logits, 
                                        )


# TODO: move to utils
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

# TODO: move to utils
def annotate_test_dataframe(pred_output):
    test_df['logits'] = pred_output.predictions[:, 1]
    test_df['pred'] = np.argmax(pred_output.predictions, 1)
    test_df['score'] = softmax(pred_output.predictions)[:, 1]

    annotated_test_data.append(test_df.copy())

# TODO: move to utils  
def compute_fold_metrics(pred_output, should_annotate_test_df=False):
    metrics = {}

    labels = pred_output.label_ids

    predictions = np.argmax(pred_output.predictions, 1)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    metrics.update({
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1})

    if should_annotate_test_df:
        annotate_test_dataframe(pred_output)

    return metrics

# TODO: move to utils
def compute_overall_metrics(data):
    overall_metrics = {}

    def get_pr_table(labels, scores):
        precs, recs, thresholds = precision_recall_curve(labels, scores)
        pr_df = pd.DataFrame({'threshold': thresholds, 'precision': precs[:-1], 'recall': recs[:-1]})
        pr_df = pr_df.sample(n=min(1000, len(pr_df)), random_state=0)
        pr_df = pr_df.sort_values(by='threshold')
        pr_table = wandb.Table(dataframe=pr_df)
        return pr_table

    scores = data['score']
    preds = data['pred'] == 1
    labels = data['label'] == 1

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    aps = average_precision_score(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    overall_metrics.update({'avg_acc': acc,
                            'avg_prec': prec,
                            'avg_rec': rec,
                            'avg_f1': f1,
                            'avg_aps': aps,
                            'avg_roc_auc': roc_auc,
                            })

    # log pr-curve
    pr_table = get_pr_table(labels, scores)
    overall_metrics.update({'pr_table': pr_table})

    return overall_metrics

# TODO: move to utils
class EvaluateCB(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = state.epoch
        metrics = {'epoch': epoch}

        train_pred_output = trainer.predict(train_dataset)
        if train_pred_output is not None:
            train_metrics = compute_fold_metrics(train_pred_output, False)
            train_metrics['loss'] = train_pred_output.metrics['test_loss']
            train_metrics = {f'train_eval/{k}': v for k, v in train_metrics.items()}

            metrics.update(train_metrics)

        dev_pred_output = trainer.predict(dev_dataset)
        if dev_pred_output is not None:
            dev_metrics = compute_fold_metrics(dev_pred_output, False)
            dev_metrics['loss'] = dev_pred_output.metrics['test_loss']
            dev_metrics = {f'dev/{k}': v for k, v in dev_metrics.items()}
            metrics.update(dev_metrics)

        test_pred_output = trainer.predict(test_dataset)
        if test_pred_output is not None:
            test_metrics = compute_fold_metrics(test_pred_output, epoch == epochs)
            test_metrics['loss'] = test_pred_output.metrics['test_loss']
            test_metrics = {f'test/{k}': v for k, v in test_metrics.items()}
            metrics.update(test_metrics)

        wandb.log(metrics)

# TODO: move to utils
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_max_len', type=int, default=128,
                        help='The maximum length of input sequences for the tokenizer.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for training.')
    parser.add_argument('--model', type=str, default="",
                        help='Type of model to use.')             
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    for seed in [0,1,2,3,4]:
        run = wandb.init(project="DL4Molecules")

        model_id_or_path = "/models/" + args.model
        wandb.config['text_model_id_or_path'] = model_id_or_path

        tokenizer_id_or_path = "/models/" + args.model
        wandb.config['tokenizer_id_or_path'] = tokenizer_id_or_path 

        tokenizer_max_len = args.tokenizer_max_len
        wandb.config['tokenizer_max_len'] = tokenizer_max_len

        epochs = args.epochs
        wandb.config['epochs'] = epochs

        wandb.config['training_seed'] = seed

        dataloader_config = {'per_device_train_batch_size': 16,
                             'per_device_eval_batch_size': 64}
        wandb.config.update(dataloader_config)

        # TODO: check if this is necessary
        #preprocessing_config = {}
        #wandb.config.update(preprocessing_config)

        learning_rate = args.learning_rate
        wandb.config['learning_rate'] = learning_rate

        num_labels = 2
        problem_type = "single_label_classification"

        wandb.config['num_labels'] = num_labels
        wandb.config['problem_type'] = problem_type

        freezed_until = ""
        wandb.config['freezed_until'] = freezed_until

        # name of the model
        # TODO: add more information about the model and hyperparameters
        model_type = args.model

        wandb.config['model_type'] = model_type

        print('load models, tokenizer and processor')
        tokenizer_config = {'pretrained_model_name_or_path': tokenizer_id_or_path,
                            'max_len': tokenizer_max_len}
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
        text_model = AutoModel.from_pretrained(model_id_or_path)
        model = ClassificationModel(text_model)

        # TODO: check if this is necessary
        if freezed_until != "":
            assert freezed_until in [n for n, _ in model.named_parameters()]
            req_grad = False
            for n, param in model.named_parameters():
                param.requires_grad = req_grad
                if freezed_until == n:
                    req_grad = True

                print(f'Parameters {n} require grad: {param.requires_grad}')

        wandb.config['n_parameters'] = count_parameters(model)

        print('load data')

        annotated_test_data = []

        training_args = TrainingArguments(
            output_dir="results",  # output directory
            num_train_epochs=epochs,  # total number of training epochs
            **dataloader_config,
            warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            learning_rate=learning_rate,
            logging_dir='./logs',  # directory for storing logs
            logging_strategy='epoch',
            save_strategy='no',
            evaluation_strategy="no",  # evaluate each `logging_steps`
            no_cuda=False,
            report_to='wandb',
            dataloader_num_workers=10,
        )

        trainer = Trainer(
            model=model,  # the instantiated Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            callbacks=[EvaluateCB]
        )

        trainer.train()
        print('***** Finished Training *****\n\n\n')

        # TODO: save model

        print('Evaluate all folds')
        data = pd.concat(annotated_test_data)
        run_name = model_type + "_" + str(seed)
        data.to_pickle(f"results/{run_name}_preds.pkl")
        metrics = compute_overall_metrics(data)
        wandb.log(metrics)
        wandb.finish()