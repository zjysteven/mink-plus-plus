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
parser.add_argument('--model', type=str, default='EleutherAI/pythia-2.8b')
parser.add_argument(
    '--dataset', type=str, default='WikiMIA_length32', 
    choices=[
        'WikiMIA_length32', 'WikiMIA_length64', 'WikiMIA_length128', 
        'WikiMIA_length32_paraphrased',
        'WikiMIA_length64_paraphrased',
        'WikiMIA_length128_paraphrased', 
    ]
)
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
    
    if 'mamba' in name:
        try:
            from transformers import MambaForCausalLM
        except ImportError:
            raise ImportError
        model = MambaForCausalLM.from_pretrained(
            name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
        )        
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

model, tokenizer = load_model(args.model)

# load dataset
if not 'paraphrased' in args.dataset:
    dataset = load_dataset('swj0419/WikiMIA', split=args.dataset)
else:
    dataset = load_dataset('zjysteven/WikiMIA_paraphrased_perturbed', split=args.dataset)
data = convert_huggingface_data_to_list_dic(dataset)

# inference - get scores for each input
scores = defaultdict(list)
for i, d in enumerate(tqdm(data, total=len(data), desc='Samples')): 
    text = d['input']
    
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    ll = -loss.item() # log-likelihood

    # assuming the score is larger for training data
    # and smaller for non-training data
    # this is why sometimes there is a negative sign in front of the score
    
    # loss and zlib
    scores['loss'].append(ll)
    scores['zlib'].append(ll / len(zlib.compress(bytes(text, 'utf-8'))))

    # mink and mink++
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
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

# compute metrics
# tpr and fpr thresholds are hard-coded
def get_metrics(scores, labels):
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05

labels = [d['label'] for d in data] # 1: training, 0: non-training
results = defaultdict(list)
for method, scores in scores.items():
    auroc, fpr95, tpr05 = get_metrics(scores, labels)
    
    results['method'].append(method)
    results['auroc'].append(f"{auroc:.1%}")
    results['fpr95'].append(f"{fpr95:.1%}")
    results['tpr05'].append(f"{tpr05:.1%}")

df = pd.DataFrame(results)
print(df)

save_root = f"results/{args.dataset}"
if not os.path.exists(save_root):
    os.makedirs(save_root)

model_id = args.model.split('/')[-1]
if os.path.isfile(os.path.join(save_root, f"{model_id}.csv")):
    df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False, mode='a', header=False)
else:
    df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False)