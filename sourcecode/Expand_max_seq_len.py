##########################################################
# Sourcecode to expand max_seq_len of LM
# (e.g. roberta-base: seq_len = 512 => 512 + PRE_SEQ_LEN)
##########################################################

import logging

import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
					
logger = logging.getLogger(__name__)

PRE_SEQ_LEN = 25

### Load pretrained model
model = RobertaForMaskedLM.from_pretrained('roberta-large')

### Update config to finetune token type embeddings
model.config.type_vocab_size = 2

### Create a new Embeddings layer, with 2 possible segments IDs instead of 1
model.roberta.embeddings.token_type_embeddings = nn.Embedding(2, model.config.hidden_size)

### Initialize it
model.roberta.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

print('before adding the tokenizer:', len(tokenizer))
tokenizer.add_special_tokens({'additional_special_tokens': ['<e>', '</e>']})
print('after adding the tokenizer:', len(tokenizer))

model.resize_token_embeddings(len(tokenizer))

# Expanding Position Embedding
pre_seq_len = PRE_SEQ_LEN
cur_max_len = model.config.max_position_embeddings
new_embeddings = nn.Embedding(cur_max_len + pre_seq_len, model.config.hidden_size)
cur_pos_embeddings = model.roberta.embeddings.position_embeddings
new_embeddings.weight.data[:cur_max_len, :] = cur_pos_embeddings.weight.data[:cur_max_len, :]

model.roberta.embeddings.position_embeddings = new_embeddings
model.roberta.embeddings.position_ids = torch.arange(cur_max_len + pre_seq_len).unsqueeze(dim=0)
model.config.max_position_embeddings = cur_max_len + pre_seq_len

tokenizer.save_pretrained('../rsc/config/roberta-large-nsp-evidence_token-pre_seq25')
model.save_pretrained('../rsc/config/roberta-large-nsp-evidence_token-pre_seq25')
