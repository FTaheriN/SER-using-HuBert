import os
import librosa
import pandas as pd

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor
from sklearn.model_selection import train_test_split


model_id = "facebook/hubert-base-ls960"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')

def get_dataframe(file_path):
    paths = []
    labels = []
    for dirname, _, filenames in os.walk(file_path):
        for filename in filenames:
            if filename.split('.')[1] == 'wav':
                paths.append(os.path.join(dirname, filename))
                labels.append(filename[3].lower())
            else:
                continue

    df = pd.DataFrame(zip(paths, labels), columns=["path", "label"])
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=12, stratify=df['label'])
    df_valid, df_test = train_test_split(df_test, test_size=0.5, random_state=12, stratify=df_test['label'])

    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test.reset_index(drop=True)


class ShEMOataset(Dataset):

    def __init__(self, df, sampling_rate: int = 16000):

        self.label_dict = {'a':0, 's':1, 'h':2, 'f':3, 'w':4, 'n':5}

        self.n_classes = len(self.label_dict)
        self.sampling_rate = sampling_rate

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.loc[idx,'path']
        audio_label = self.label_dict[self.df.loc[idx,'label']]

        audio_array, rate = librosa.load(audio_path, sr=16000, duration=7.0)
        return audio_array, audio_label


def collate_batch(batch):
    audio_list, labels = [], []

    for (audio, label) in batch:
        audio_list.append(audio)
        labels.append(label)

    audio_features = feature_extractor(audio_list,
                                       sampling_rate=16000,
                                       padding=True,
                                       return_attention_mask=True,
                                       return_tensors="pt")
                                      #  


    return audio_features.input_values, audio_features.attention_mask, torch.tensor(labels) #
 
def get_dataloader(file_path, batch_size=32):
    train_df, valid_df, test_df = get_dataframe(file_path)

    train_dataset = ShEMOataset(train_df)
    valid_dataset = ShEMOataset(valid_df)
    test_dataset  = ShEMOataset(test_df )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader  = DataLoader(test_dataset , batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    return train_loader, valid_loader, test_loader
