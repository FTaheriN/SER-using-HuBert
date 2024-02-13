import torch
from torch import nn
import torch.nn.functional as F
from transformers import HubertModel, HubertConfig
from typing import Union



class HuBertWithNNClassifier(nn.Module):
    
    def __init__(self):

        super(HuBertWithNNClassifier, self).__init__()

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



    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

        
    def forward(self, audio_features, attention_mask):
        output = self.hubert(audio_features, attention_mask)

        hidden_states = output.last_hidden_state #[:,0]

        hidden_states = self.linear1(hidden_states)

        padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        hidden_states[~padding_mask] = 0.0
        pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        out = self.relu(self.batch_norm1(pooled_output))
        out = self.linear2(self.dropout2(out))

        return out