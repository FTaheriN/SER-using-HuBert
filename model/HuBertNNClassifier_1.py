import torch
from torch import nn
import torch.nn.functional as F
from transformers import HubertModel, HubertConfig
from typing import Union



class HuBertWithNNClassifier_1(nn.Module):
    
    def __init__(self):

        super(HuBertWithNNClassifier_1, self).__init__()

        self.hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960')
        self.config = HubertConfig.from_pretrained('facebook/hubert-base-ls960')

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()

        self.batch_norm0 = nn.BatchNorm1d(768)
        self.batch_norm1 = nn.BatchNorm1d(1024)
      
        self.linear1 = nn.Linear(768, 1024)
        self.linear2 = nn.Linear(1024, 6)

        self.hubert.feature_extractor._freeze_parameters()

        
    def forward(self, audio_features, attention_mask):
        output = self.hubert(audio_features, attention_mask)

        hidden_states = output.last_hidden_state #[:,0]

        out = self.batch_norm0(hidden_states.mean(dim=1))

        out = self.dropout1(out)

        out = self.linear1(out) 
        out = self.relu(self.batch_norm1(out))
        out = self.linear2(self.dropout2(out))

        return out