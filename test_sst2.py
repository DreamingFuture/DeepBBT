"""
Given trained prompt and corresponding template, we ask test API for predictions
Shallow version
We train one prompt after embedding layer, so we use sentence_fn and embedding_and_attention_mask_fn.
Baseline code is in bbt.py
"""


import os
import torch
from transformers import RobertaConfig, RobertaTokenizer
from models.modeling_roberta import RobertaModel, RobertaForMaskedLM
import numpy as np
import csv
from dataloader import SST2Loader
from metrics import SST2Metric, AGNewsMetric, YelpPMetric, MRPCMetric, SNLIMetric, TRECMetric
from sklearn.metrics import f1_score
from utils import hinge_loss
import copy
from fastNLP import Tester

cat_or_add = 'add'
intrinsic_dim = 500
random_proj = 'normal'
sigma = 1

class LMForwardAPI:
    def __init__(self, model_name='roberta-large', n_prompt_tokens=50, task_name='sst2', loss_type='hinge', init_prompt_path=None):
        if model_name in ['roberta-large']:
            self.config = RobertaConfig.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
                inference_framework='pt',
                onnx_model_path=None,
            )
            self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))
        else:
            raise NotImplementedError

        if cat_or_add == 'cat':
            self.model.set_concat_prompt(True)
            if init_prompt_path is not None:
                print('Initialize prompt embedding from {}'.format(init_prompt_path))
                self.init_prompt = torch.load(init_prompt_path).weight.cpu().reshape(-1)
            else:
                print('Initial prompt embedding not found. Initialize to zero embedding.')
                self.init_prompt = torch.zeros(n_prompt_tokens * self.config.hidden_size)
            print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))
        else:
            # self.model.set_concat_prompt(False)
            self.init_prompt = None
        self.model.to(device)
        self.model.eval()
        self.linear = torch.nn.Linear(intrinsic_dim, n_prompt_tokens * self.config.hidden_size, bias=False)
        if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['roberta-base', 'roberta-large']:
                embedding = self.model.roberta.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
                embedding = self.model.bert.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['google/electra-base-generator', 'google/electra-large-generator']:
                embedding = self.model.electra.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['facebook/bart-base', 'facebook/bart-large', 'fnlp/cpt-large']:
                embedding = self.model.model.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                embedding = self.model.transformer.get_input_embeddings().weight.clone().cpu()
            else:  # T5
                embedding = self.model.get_input_embeddings().weight.clone().cpu()
            # embedding = embedding[1000: 2000]
            mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = std_hat / (np.sqrt(intrinsic_dim) * sigma)
            # temp = intrinsic_dim - std_hat * std_hat
            # mu = mu_hat / temp
            # std = std_hat / np.sqrt(temp)
            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():
                torch.nn.init.normal_(p, mu, std)
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_prompt = None
        self.num_call = 0
        self.loss_type = loss_type
        if task_name == 'sst2':
            self.metric = SST2Metric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SST2Metric'
        elif task_name == 'agnews':
            self.metric = AGNewsMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'AGNewsMetric'
        elif task_name == 'yelpp':
            self.metric = YelpPMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'YelpPMetric'
        elif task_name == 'mrpc':
            self.metric = MRPCMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'MRPCMetric'
        elif task_name == 'snli':
            self.metric = SNLIMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SNLIMetric'
        elif task_name == 'trec':
            self.metric = TRECMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'TRECMetric'
        else:
            raise NotImplementedError
        self.margin = self.metric.margin
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def calc_metric(self, logits, target):
        label_map = self.metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        interest_index = list(label_map.keys())
        logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)

        if self.metric_key == 'acc':
            perf = (pred == converted_target).sum() / len(target)
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
                            pred.detach().cpu().numpy().tolist(), average='macro')
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')

        if self.loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
        elif self.loss_type == 'ce':
            loss = self.ce_loss(logits, converted_target).item()
        elif self.loss_type == 'perf':
            loss = -1 * perf
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

        return loss, perf, logits

    def eval(self, prompt_embedding=None, test_data=None):
        bsz = len(test_data['input_ids'])  # batch size of dev data is the orignal batch size of training data
        if self.init_prompt is not None:
            prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0

        prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1).repeat(bsz, 1, 1)
        self.model.set_prompt_embedding(prompt_embedding)

        for k, v in test_data.items():
            test_data[k] = v.to(device)
        with torch.no_grad():
            logits = self.model(
                input_ids=test_data['input_ids'],
                attention_mask=test_data['attention_mask'],
                mask_pos=test_data['mask_pos'],
            )['logits']

        dev_loss, dev_perf, logits= self.calc_metric(logits, test_data['labels'])
        for sample_idx in range(bsz):
            logit = logits[sample_idx]
            pred = logits[sample_idx]
            print(10 * '-')
            print(logit)
            if int(pred.argmax(dim=-1)) == 0:
                print('bad')
            else:
                print('great')
            print('label:', tokenizer.convert_ids_to_tokens(int(test_data['labels'][sample_idx])))
        return dev_perf


task_name = 'sst2'
device = 'cuda:3'
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large').to(device).eval()

pre_str = tokenizer.decode(list(range(1000, 1050))) + ' . '
post_str = ' . It was <mask> .'
n_prompt_tokens = 50
DataLoader = SST2Loader

model_forward_api = LMForwardAPI(task_name=task_name)

for seed in [8, 13, 42, 50, 60]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    best = torch.load(f'./results/{task_name}/{seed}/best.pt').to(device).view(50, -1)

    data_bundle = DataLoader(tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens).my_load(['dev'], seed)
    dev_data = data_bundle.get_dataset('dev')
    dev_data.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    dev_data.set_pad_val('attention_mask', 0)

    dev_data = {
        'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
        'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
        'mask_pos': torch.tensor(dev_data['mask_pos'].get(list(range(len(dev_data))))),
        'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
    }


    predictions = torch.tensor([], device=device)
    acc = model_forward_api.eval(prompt_embedding=best, test_data=dev_data)
    print(seed, acc)
    # exit(0)
    #
    # if not os.path.exists(f'./predictions/{task_name}'):
    #     os.makedirs(f'./predictions/{task_name}')
    # with open(f'./predictions/{task_name}/{seed}.csv', 'a+') as f:
    #     wt = csv.writer(f)
    #     wt.writerow(['', 'pred'])
    #     wt.writerows(torch.stack([torch.arange(predictions.size(0)), predictions.detach().cpu()]).long().T.numpy().tolist())





