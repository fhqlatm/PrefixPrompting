import os
import sys
import json
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

class DatasetCustom(Dataset): 
    def __init__(self, path):
        with open(path, 'r') as f:
            self.x = json.load(f)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

def make_label_to_tensor(labels):
    batch_labels = []
    
    for label in labels:
        sample_label = [label]
        batch_labels.append(sample_label)
        
    return torch.tensor(batch_labels)

def custom_collate_fn(tokenizer: AutoTokenizer, batch):

    input_sentences = [sample[0] for sample in batch]
    input_labels = [sample[1] for sample in batch]

    batch_inputs = tokenizer(input_sentences, padding = True, return_tensors = "pt")
    batch_labels = make_label_to_tensor(input_labels)

    return batch_inputs, batch_labels