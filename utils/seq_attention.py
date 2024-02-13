import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer

import seaborn as sns
import matplotlib.pyplot as plt




def plot_attention(test_df):
    
    mbert = 'bert-base-multilingual-uncased'

    eg = test_df.loc[test_df['p_lengths'] == 8].iloc[-2,:]
    text1 = eg['premise']
    text2 = eg['hypothesis']

    tokenizer =  AutoTokenizer.from_pretrained(mbert)
    tok1 = tokenizer.tokenize(text1)
    tok2 = tokenizer.tokenize(text2)

    tok =['[CLS]'] + tok1 + ['[SEP]'] + tok2 +['[SEP]']

    # sent1_token_codes + [tokenizer.convert_tokens_to_ids('[SEP]')] + sent2_token_codes + [tokenizer.convert_tokens_to_ids('[SEP]')] 

    model = torch.load('/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/checkpoints/mbert_relu1.pth')

    ids = torch.tensor(tokenizer.convert_tokens_to_ids(tok)).unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model.bert(ids, None, None)
    attentions = torch.cat(output[3]).to('cpu')

    attentions_pos = attentions[11, :] #7:11

    cols = 2
    rows = 6

    fig, axes = plt.subplots( rows,cols, figsize = (10,40))
    axes = axes.flat

    for i,att in enumerate(attentions_pos):

        sns.heatmap(att,vmin = 0, vmax = 1,ax = axes[i], xticklabels = tok, yticklabels = tok)
        axes[i].set_title(f'head - {12-i} ' )
    fig.tight_layout()