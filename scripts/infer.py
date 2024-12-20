# -*- coding: utf-8 -*-

import torch
import time
import numpy as np

from transformers import BertTokenizer
from scripts.config import parse_args
from tqdm import tqdm
from models.bert import BERT_Model
import json
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    example = [lemmatize_words(example)]
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)

class bert_inferencing_sentence():

    def __init__(self, device, logger):

        self.opts = parse_args()
        self.device = device
        self.logger = logger
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.opts.bert_config, do_lower_case=True)
        
        self.model = BERT_Model(
            bert_config=self.opts.bert_config,
            n_class=self.opts.n_class
        )
        self.model.load_state_dict(torch.load(self.opts.modelDir, map_location=self.device)['model_dict'])
        self.model.to(device)
        self.logger.info("Bert model loading is done")

    def process(self, text_data):
        results, conf_l = [], []

        X_test = convert_lines(text_data, self.opts.max_length, self.bert_tokenizer)
        test_data = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
        test_loader = torch.utils.data.DataLoader(test_data, 
                        batch_size = self.opts.batch_size, shuffle=False)

        for _, (x_batch,) in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                tic = time.time()
                
                pred = self.model(x_batch.to(self.device), 
                            mask=(x_batch > 0).to(self.device), 
                            token_type_ids=None)
                
                pred_prob = torch.softmax(pred, dim=1).cpu()
                probs = torch.nn.functional.softmax(pred, dim=1)
                conf, _ = torch.max(probs, 1)
                conf_l.append(list(conf.tolist()))
                results.append(list(pred_prob.argmax(-1).tolist()))
                torch.cuda.empty_cache()
                self.logger.info("Time taken by Bert Model is {}".format(time.time() -tic))

        conf_l = list(np.concatenate(conf_l))
        results = list(np.concatenate(results))
        predicted = [*map(self.opts.id2label.get, results)] 
        predicted = [ f'{x} and Prob. {y:.3f}' for x,y in zip(predicted, conf_l) ]
        predicted = {int(i+1): v for i, v in enumerate(predicted)}
        
        output = {"BERT": predicted}
        output = json.dumps(output, cls=NpEncoder)

        return  output, predicted, text_data
    

