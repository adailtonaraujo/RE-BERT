# Extract software requirements from IOB RE-BERT classifier

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import pickle
import nltk
from tqdm import tqdm
nltk.download('punkt')

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, pad_and_truncate
from models.re_bert import RE_BERT
from models.lcf_bert import LCF_BERT

from transformers import BertModel

class Inferer:

    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)

        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text, aspect):
        aspect = aspect.lower().strip()
        text_left, _, text_right = [s.strip() for s in text.lower().partition(aspect)]
        
        text_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
        context_indices = self.tokenizer.text_to_sequence(text_left + " " + text_right)
        left_indices = self.tokenizer.text_to_sequence(text_left)
        left_with_aspect_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect)
        right_indices = self.tokenizer.text_to_sequence(text_right, reverse=True)
        right_with_aspect_indices = self.tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
        aspect_indices = self.tokenizer.text_to_sequence(aspect)
        left_len = np.sum(left_indices != 0)
        aspect_len = np.sum(aspect_indices != 0)
        aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

        text_len = np.sum(text_indices != 0)
        concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

        text_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
        aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")


        data = {
            'concat_bert_indices': concat_bert_indices,
            'concat_segments_indices': concat_segments_indices,
            'text_bert_indices': text_bert_indices,
            'aspect_bert_indices': aspect_bert_indices,
            'text_indices': text_indices,
            'context_indices': context_indices,
            'left_indices': left_indices,
            'left_with_aspect_indices': left_with_aspect_indices,
            'right_indices': right_indices,
            'right_with_aspect_indices': right_with_aspect_indices,
            'aspect_indices': aspect_indices,
            'aspect_boundary': aspect_boundary
        }

        t_inputs = [torch.tensor([data[col]], device=self.opt.device) for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()

        return t_probs


def get_classifier(opt):
    model_classes = {
        'RE_BERT': RE_BERT,
    }
    input_colses = {
        'RE_BERT': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    }

    opt.model_name = 'RE_BERT'
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    # set your trained models here
    opt.state_dict_path = opt.classifier_model_file

    inf = Inferer(opt)

    return inf

def fit(classifier, review=['left text', 'requirement', 'right text.']):
    txt_review = review[0]+' '+review[1]+' '+review[2]
    txt_requirement = review[1]
    t_probs = classifier.evaluate(txt_review, txt_requirement)
    result = {}
    result['confidences'] = list(t_probs[0])
    result['iob'] = (t_probs.argmax(axis=-1)-1)[0]
    
    return result

def get_iob(classifier,review_sentence):
  doc = nltk.word_tokenize(review_sentence)

  tokens = []
  for token in doc: tokens.append(str(token))

  results = []

  for i in range(0,len(tokens)):
    aspect_candidate = ''
    left = ''
    right = ''
    for j in range(0,len(tokens)):
      if i==j: aspect_candidate = tokens[j]
      if j < i: left += tokens[j]+' '
      if j > i: right += tokens[j]+' '

    review = [left,aspect_candidate,right]

    output = fit(classifier, review)
    results.append([aspect_candidate,output])
  
  return results

def extract(classifier, review):

    sent_text = nltk.sent_tokenize(review)
    sentences = []
    for sentence in sent_text:
        sentences.append(sentence)

    extracted_data = []
    for sentence in tqdm(sentences,desc="Extract software requirements candidates"):
      sent=sentence.strip()
      results = get_iob(classifier,sent)
      requirements_candidates = []
      for item in results:
        if item[1]['iob']!=-1: requirements_candidates.append(item[0])
      extracted_data.append([sent,requirements_candidates,results])
    
    return extracted_data

def re_bert_model(options):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='RE_BERT', type=str)
    parser.add_argument('--classifier_model_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--dataset', default='API_MODE', type=str, help='app_name')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=1, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--local_context_focus', default='cdm', type=str)
    parser.add_argument('--alpha', default=3, type=int, help='relative distance (LOCAL CONTEXT)')
    opt = parser.parse_args(options)
    opt.SRD = opt.alpha

    input_colses = {
        'RE_BERT': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    model_classes = {
        'RE_BERT': RE_BERT,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    classifier = get_classifier(opt)

    return classifier
