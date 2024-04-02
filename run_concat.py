import os, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# helper function
def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='huggyllama/llama-7b')
parser.add_argument('--half', action='store_true')
parser.add_argument('--int8', action='store_true')
args = parser.parse_args()

# load model
def load_model(name):
    int8_kwargs = {}
    half_kwargs = {}
    if args.int8:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif args.half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

model, tokenizer = load_model(args.model)

# load dataset
dataset = load_dataset('zjysteven/WikiMIA_concat', split='WikiMIA_concat')
data = convert_huggingface_data_to_list_dic(dataset)
labels = [d['label'] for d in data]

# inference - get scores for each input
scores = defaultdict(list)
chunk_labels = []
for i, d in enumerate(tqdm(data, total=len(data), desc='Samples')): 
    full_text = d['input']
    assert len(full_text.split(' ')) // 32 == len(labels[i]), \
        f"{i}, {len(full_text.split(' '))} != {len(labels[i])}"
    tmp = full_text.split(' ')
    text_chunks = [' '.join(tmp[j*32:(j+1)*32]) for j in range(len(labels[i]))]

    # inference
    input_ids = torch.tensor(tokenizer.encode(full_text)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    # locate each chunk's token ids and sanity check
    chunk_token_lens = []
    for j, chunk in enumerate(text_chunks):
        tmp_ids = tokenizer.encode(chunk)
        if j > 0:
            chunk_token_lens.append(len(tmp_ids) - 1)
        else:
            chunk_token_lens.append(len(tmp_ids))
    assert sum(chunk_token_lens) == len(input_ids[0]), \
        f"{i}, {sum(chunk_token_lens)} != {len(input_ids[0])}"

    # calculate each chunk's scores
    for j, chunk in enumerate(text_chunks):
        chunk_input_ids = input_ids[0][sum(chunk_token_lens[:j]):sum(chunk_token_lens[:j+1])]
        chunk_logits = logits[0][sum(chunk_token_lens[:j]):sum(chunk_token_lens[:j+1])]

        chunk_input_ids = chunk_input_ids[1:].unsqueeze(-1)
        chunk_logits = chunk_logits[:-1]

        loss = F.cross_entropy(
            chunk_logits.unsqueeze(0).permute(0, 2, 1), 
            chunk_input_ids[:, 0].unsqueeze(0)
        )
        ll = -loss.item() # log-likelihood

        # assuming the score is larger for training data
        # and smaller for non-training data
        # this is why sometimes there is a negative sign in front of the score
        scores['loss'].append(ll)
        scores['zlib'].append(
            ll / len(zlib.compress(bytes(chunk, 'utf-8')))
        )

        # mink and mink++
        probs = F.softmax(chunk_logits, dim=-1)
        log_probs = F.log_softmax(chunk_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=chunk_input_ids).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        ## mink
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.cpu())[:k_length]
            scores[f'mink_{ratio}'].append(np.mean(topk).item())
            
        ## mink++
        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            scores[f'mink++_{ratio}'].append(np.mean(topk).item())

        chunk_labels.append(labels[i][j])
    
# compute metrics
# tpr and fpr thresholds are hard-coded
def get_metrics(scores, labels):
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05

results = defaultdict(list)
for method, scores in scores.items():
    auroc, fpr95, tpr05 = get_metrics(scores, chunk_labels)
    
    results['method'].append(method)
    results['auroc'].append(f"{auroc:.1%}")
    results['fpr95'].append(f"{fpr95:.1%}")
    results['tpr05'].append(f"{tpr05:.1%}")

df = pd.DataFrame(results)
print(df)

save_root = f"results/WikiMIA_concat"
if not os.path.exists(save_root):
    os.makedirs(save_root)

model_id = args.model.split('/')[-1]
if os.path.isfile(os.path.join(save_root, f"{model_id}.csv")):
    df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False, mode='a', header=False)
else:
    df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False)