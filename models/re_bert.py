# The code is based on LCF-BERT (repository: https://github.com/yangheng95/LCF-ABSA)

import torch
import torch.nn as nn
import copy
import numpy as np

from transformers.modeling_bert import BertPooler, BertSelfAttention
from models.lcf_bert import LCF_BERT

class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class RE_BERT(LCF_BERT):
    def __init__(self, bert, opt):
        super(LCF_BERT, self).__init__()

        self.bert_spc = bert
        self.opt = opt
        self.bert_local = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
