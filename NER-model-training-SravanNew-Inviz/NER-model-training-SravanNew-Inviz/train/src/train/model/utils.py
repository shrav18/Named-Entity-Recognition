from typing import List

import torch
import torch.nn as nn
from transformers import AlbertTokenizerFast, AlbertModel, AlbertConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.preprocessing import normalize
import numpy as np
import itertools
from collections.abc import Iterable
import joblib
import numpy as np
import pandas as pd
import os
import csv
from tqdm import tqdm
from src.schema.schema import JobArgs

#from tqdm.auto import tqdm
#tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EntityDataset:
    def __init__(self, texts,  tags,job_args: JobArgs,TOKENIZER,other_token):
        self.texts = texts
        self.tags = tags
        self.job_args= job_args
        self.TOKENIZER = TOKENIZER
        self.other_token = other_token
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        target_tag =[]

        for i, s in enumerate(text):
            inputs = self.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:self.job_args.model_configs.training_params.max_len - 2]
        target_tag = target_tag[:self.job_args.model_configs.training_params.max_len - 2]

        ids = [2] + ids + [3]
        target_tag = [self.other_token] + target_tag + [self.other_token]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = self.job_args.model_configs.training_params.max_len - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([self.other_token] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }

class InferenceDataset:
    def __init__(self, texts, job_args: JobArgs,TOKENIZER):
        self.texts = texts
        self.job_args: job_args
        self.TOKENIZER = TOKENIZER

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]

        ids = []

        for i, s in enumerate(text):
            inputs = self.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)

        ids = ids[:self.job_args.model_configs.training_params.max_len- 2]

        ids = [2] + ids + [3]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = self.job_args.model_configs.training_params.max_len- len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
        }


def find_delimiter(filename) -> str:
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return str(delimiter)


def process_data(data_paths: List[str], job_args: JobArgs):
    output_path = "./"
    df = None
    for data_path in data_paths:
        df_ = pd.read_csv(data_path, sep=find_delimiter(data_path))
        if df is not None:
            df_ = df_[~df_['query'].isin(df['query'])]
            df = df.append(df_, ignore_index=True)
        else:
            df = df_

    df = df.dropna(subset=['query', 'word', 'label'])
    df.loc[:, "query"] = df["query"]
    if job_args.model_configs.incremental:
        enc_tag = joblib.load(output_path + "ner_enc_tag.bin")
        new_labels = len(set(df["label"]) - set(enc_tag.classes_))
        if new_labels > 0:
            raise Exception("New labels in dataset are not supported: " + str(new_labels))
        df.loc[:, "tag"] = enc_tag.transform(df["label"])
    else:
        enc_tag = preprocessing.LabelEncoder()
        df.loc[:, "tag"] = enc_tag.fit_transform(df["label"])

    df = df.groupby('query',as_index = False).aggregate({'word':(lambda x: list(x)),'label':(lambda x: list(x)),'tag':(lambda x: list(x))})
#     sentences = df["word"].values
#     tag = df["label"].values
    return df, enc_tag

def flatten(data):
    for x in data:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def metrics_(y,y_):
    
    y_true = np.array(list(flatten(y)))
    y_pred = np.array(list(flatten(y_)))

    y_indexs = set(list(np.where(y_true == 1)[0]) + list(np.where(y_pred > 0.5 )[0]))
    
    y_true = [y_true[i] for i in y_indexs]
    y_pred = [y_pred[i] for i in y_indexs]
    
    score = {
        'batch recall':recall_score(y_true, y_pred, average='micro'),
        'batch precision':precision_score(y_true, y_pred, average='micro'),
        'batch F1 flat':f1_score(y_true, y_pred, average='micro'),
    }
    print(score)
    return score
    
