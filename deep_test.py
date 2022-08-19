"""
Given trained prompt and corresponding template, we ask test API for predictions
Shallow version
We train one prompt after embedding layer, so we use sentence_fn and embedding_and_attention_mask_fn.
Baseline code is in bbt.py
"""


import argparse
import os
import torch
from test_api import test_api
from test_api import RobertaEmbeddings
from transformers import RobertaConfig, RobertaTokenizer
from models.modeling_roberta import RobertaModel
import numpy as np
import csv


task_name_dict = {
    'sst2': 'SST-2',
    'agnews': 'AGNews',
    'yelpp': 'Yelp',
    'mrpc': 'MRPC',
    'snli': 'SNLI',
    'trec': 'TREC'
}
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='sst2', type=str)
parser.add_argument('--device', default='cuda:6', type=str)
parser.add_argument('--batch_size', default=12, type=int)
args = parser.parse_args()

task_name = args.task_name
device = args.device
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large').to(device).eval()

pre_str = tokenizer.decode(list(range(1000, 1050))) + ' . '
middle_str = '? <mask> .'

if task_name in ['sst2', 'yelpp']:
    middle_str = "It is <mask>."       
elif task_name in ['mrpc']:
    middle_str = '? <mask> .'
elif task_name in ['trec']:
    middle_str = 'The kind of news is <mask> .'
elif task_name in ['snli']:
    middle_str = '<mask> , '
elif task_name in ['agnews']:
    middle_str = 'The kind of news is <mask> .'
else:
    raise ValueError

for seed in [8, 13, 42, 50, 60]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # best = torch.load(f'./results_bbt/{task_name}/{seed}/best.pt').to(device).view(50, -1)
    # deepbbt
    best = torch.load(f'./results/{task_name}/{seed}/best.pt').to(device)

    def sentence_fn(test_data):
        """
        This func can be a little confusing.
        Since there are 2 sentences in MRPC and SNLI each sample, we use the same variable `test_data` to represent both.
        test_data is actually a <dummy_token>. It is then replaced by real data in the wrapped API.
        For other 4 tasks, test_data must be used only once, e.g. pre_str + test_data + post_str
        """
        if task_name in ['mrpc', 'snli']:
            return pre_str + test_data + middle_str + test_data
        else:
            return pre_str + test_data + middle_str


    def embedding_and_attention_mask_fn(embedding, attention_mask):
        # res = torch.cat([init_prompt[:-5, :], input_embed, init_prompt[-5:, :]], dim=0)
        prepad = torch.zeros(size=(1, 1024), device=device)
        pospad = torch.zeros(size=(embedding.size(1) - 51, 1024), device=device)
        return embedding + torch.cat([prepad, best, pospad]), attention_mask

    def hidden_states_and_attention_mask_fn(i, embedding, attention_mask):
        prepad = torch.zeros(size=(1, 1024), device=device)
        pospad = torch.zeros(size=(embedding.size(1) - 51, 1024), device=device)
        return embedding + torch.cat([prepad, best[i], pospad]), attention_mask

    predictions = torch.tensor([], device=device)
    # bbt
    # for res, _ in test_api(
    #     sentence_fn=sentence_fn,
    #     embedding_and_attention_mask_fn=embedding_and_attention_mask_fn,
    #     test_data_path=f'./test_datasets/{task_name_dict[task_name]}/encrypted.pth',
    #     task_name=task_name_dict[task_name],
    #     batch_size = args.batch_size,
    #     device = args.device
    # ):
    
    # deepbbt
    for res, _ in test_api(
        sentence_fn=sentence_fn,
        hidden_states_and_attention_mask_fn=hidden_states_and_attention_mask_fn,
        test_data_path=f'./test_datasets/{task_name_dict[task_name]}/encrypted.pth',
        task_name=task_name_dict[task_name],
        batch_size = args.batch_size,
        device = args.device
    ):
        if task_name in ['sst2', 'yelpp']:
            c0 = res[:, tokenizer.encode("bad", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("great", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1]).argmax(dim=0)
        elif task_name in ['mrpc']:
            c0 = res[:, tokenizer.encode("No", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("Yes", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1]).argmax(dim=0)
        elif task_name in ['trec']:
            c0 = res[:, tokenizer.encode("description", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("entity", add_special_tokens=False)[0]]
            c2 = res[:, tokenizer.encode("abbreviation", add_special_tokens=False)[0]]
            c3 = res[:, tokenizer.encode("human", add_special_tokens=False)[0]]
            c4 = res[:, tokenizer.encode("numeric", add_special_tokens=False)[0]]
            c5 = res[:, tokenizer.encode("location", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1, c2, c3, c4, c5]).argmax(dim=0)
        elif task_name in ['snli']:
            c0 = res[:, tokenizer.encode("Yes", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("Maybe", add_special_tokens=False)[0]]
            c2 = res[:, tokenizer.encode("No", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1, c2]).argmax(dim=0)
        elif task_name in ['agnews']:
            c0 = res[:, tokenizer.encode("World", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("Sports", add_special_tokens=False)[0]]
            c2 = res[:, tokenizer.encode("Business", add_special_tokens=False)[0]]
            c3 = res[:, tokenizer.encode("Tech", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1, c2, c3]).argmax(dim=0)
        else:
            raise ValueError
        
        predictions = torch.cat([predictions, pred])
    if not os.path.exists(f'./predictions/{task_name}'):
        os.makedirs(f'./predictions/{task_name}')
    with open(f'./predictions/{task_name}/{seed}.csv', 'w') as f:
        wt = csv.writer(f)
        wt.writerow(['', 'pred'])
        wt.writerows(torch.stack([torch.arange(predictions.size(0)), predictions.detach().cpu()]).long().T.numpy().tolist())





