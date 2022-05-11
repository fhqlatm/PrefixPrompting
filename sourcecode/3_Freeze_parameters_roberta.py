import os
import sys
import json
import time
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, AutoConfig
from PrefixRobertaModel import RobertaForCL
import DatasetCustom
from sklearn.metrics import classification_report

class Model(nn.Module):
    def __init__(self, roberta):
        super().__init__()
        self.roberta = roberta
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, **kargs):
        output = self.roberta(**kargs).pooler_output
        output = self.linear(output)
        output = self.sigmoid(output)
        return output

PATH_DIR = '../data/nsmc_data'
PATH_FILE_TRAIN = os.path.join(PATH_DIR, 'nsmc_train.json')
PATH_FILE_TEST = os.path.join(PATH_DIR, 'nsmc_test.json')

BATCH_SIZE = 256

PRETAINED_PREFIX_MODEL_PATH = 'klue/roberta-base'
PATH_FILE_REPORT = '../data/Freeze_roberta_report.txt'

roberta = AutoModel.from_pretrained(PRETAINED_PREFIX_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(PRETAINED_PREFIX_MODEL_PATH)
config = AutoConfig.from_pretrained(PRETAINED_PREFIX_MODEL_PATH)

dataset_train = DatasetCustom.DatasetCustom(PATH_FILE_TRAIN)
dataset_test = DatasetCustom.DatasetCustom(PATH_FILE_TEST)

partial_collate_fn = partial(DatasetCustom.custom_collate_fn, tokenizer)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=partial_collate_fn
)

dataloader_test = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial_collate_fn
)

for param in roberta.parameters():
    param.requires_grad = False

model = Model(roberta)

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)

device = torch.device('cuda:0')
model.to(device)
# model = nn.DataParallel(model, device_ids = [0,1,2,3])

CELoss = nn.BCELoss()
optimizer = AdamW(model.parameters(), lr=1.0e-5)

model.train()

epochs = 5

for epoch in range(epochs):
    start=time.time()
    train_loss = 0

    for batch in dataloader_train:
        batch_inputs = {k: v.cuda(device) for k, v in list(batch[0].items())}
        batch_labels = batch[1].cuda(device)

        output = model(**batch_inputs)

        loss = CELoss(output.view(-1).to(torch.float32), batch_labels.view(-1).to(torch.float32))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        train_loss += loss.item()/len(dataloader_train)
        
    TIME = time.time() - start
    print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s    ETA : {TIME*(epochs-epoch-1):.0f}s')
    print(f'TRAIN    loss : {train_loss:.5f}')

model.eval()

gold_list = []
pred_list = []

with torch.no_grad():
    for batch in dataloader_test:
        batch_inputs = {k: v.cuda(device) for k, v in list(batch[0].items())}
        batch_labels = batch[1].cuda(device)
        
        output = model(**batch_inputs)
        loss = CELoss(output.view(-1).to(torch.float32), batch_labels.view(-1).to(torch.float32))
        
        for gold, pred in zip(batch_labels, output):
            pred = torch.round(pred)

            gold_list.append(gold)
            pred_list.append(pred)

gold_list_flat = []
pred_list_flat = []
for g, p in zip(gold_list, pred_list):
    gold_list_flat.append(g.item())
    pred_list_flat.append(p.item())

report = classification_report(gold_list_flat, pred_list_flat, digits = 4)

with open(PATH_FILE_REPORT, "w") as f:
    print(report, file=f)