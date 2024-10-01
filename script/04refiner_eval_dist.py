import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, set_seed, DataCollatorForSeq2Seq, DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback, BitsAndBytesConfig
import csv
from datasets import Dataset
import random
from torch import cuda
from tqdm import tqdm
import time
import torch.nn as nn
import argparse
import json
import os
import evaluate
import shutil
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, recall_score
import re
import spacy
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class EvalDistinct:

    def __init__(self,  dataset_name, model_val):
        

        self.model_val = model_val
        self.dataset_name = dataset_name


        self.llama3_data =  None
        self.llama3_refine_data =  None
        self.gpt_data =  None
        self.real_data = None 
        self.llama3_supergen_data = None
  
        self.domain_map_test = None
        self.intent_map_test = None 
        self.device = 'cuda' if cuda.is_available() else 'cpu'


    def load_test_domain_intent_map(self, domain_map_filepath):
        with open(domain_map_filepath, 'r') as file:
            domain_map = json.load(file)
        intent_list_train = list(self.model_val.config.label2id.keys())
        intent_map_test = dict()
        domain_map_test = dict()
        for dm in domain_map:
            intent_list = domain_map[dm]
            for intent in intent_list:
                if intent not in intent_list_train:
                    intent_map_test[intent] = dm
                    if dm not in domain_map_test:
                        domain_map_test[dm] = [intent]
                    else:
                        domain_map_test[dm].append(intent)
        
        
        print(f'Test domain is : {list(domain_map_test.keys())}')
        print(f'Total test intent number : {len(intent_map_test.keys())}')        
        

        self.domain_map_test = domain_map_test
        self.intent_map_test = intent_map_test 
        
    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file) 
        return data       
    @staticmethod       
    def documentize(data): 
        result = dict()
        for intent in data:
            ut_list = data[intent]
            doc = ' '.join(ut_list)
            result[intent] = doc
        return result
    
    
          
    def load_csv_to_dic(self, filepath): 
        result = dict()

        with open(filepath, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            
            # Read each row from the CSV file and append it to the list_of_lists
            for row in csvreader:
                ut = row[0]
                intent = row[1]
                if intent in self.intent_map_test:
                    if intent not in result:
                        result[intent] = [ut]
                    else:
                        result[intent].append(ut)
        return result

     
    def load_all_data(self,
                      real_data_filepath,
                      gpt_data_filepath, 
                      llama3_data_filepath,
                      llama3_refine_data_filepath,
                      llama3_supergen_data_filepath):
        
        real_data = self.load_csv_to_dic(real_data_filepath)
                    
        if self.dataset_name == 'sgd':
            default_len = 200
            real_data = {intent: ut_list if len(ut_list) <= default_len  else random.sample(ut_list, default_len) for intent, ut_list in real_data.items()}
        elif self.dataset_name == 'clinc150':
            default_len = 100
            real_data = {intent: ut_list if len(ut_list) <= default_len  else random.sample(ut_list, default_len) for intent, ut_list in real_data.items()}
        else:
            raise ValueError("dataset_name value not qualified")                 
        
        
        
        gen_data = self.load_csv_to_dic(llama3_data_filepath)
        
        for intent, ut in gen_data.items():
            if self.dataset_name == 'sgd':
                gen_data = {intent: ut_list[:200] for intent, ut_list in gen_data.items()}

            elif self.dataset_name == 'clinc150':
                        gen_data = {intent: ut_list[:100] for intent, ut_list in gen_data.items()}

            else:
                raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        
        self.real_data = real_data
        self.llama3_data =  gen_data
        self.gpt_data =  EvalDistinct.load_json(gpt_data_filepath)
        self.llama3_refine_data =  EvalDistinct.load_json(llama3_refine_data_filepath)
        self.llama3_supergen_data = EvalDistinct.load_json(llama3_supergen_data_filepath)
        for intent in self.real_data:
            length = len(self.real_data[intent])
            len1 = len(self.llama3_data[intent])
            len2 = len(self.gpt_data[intent])
            len3 = len(self.llama3_refine_data[intent])
            len4 = len(self.llama3_supergen_data[intent])
            check = set([length, len1, len2, len3, len4, len5])
            len_check = len(check) 
            if len_check != 1:
                print(f'intent: {intent}, differnet length set: {check}')
            if self.dataset_name == 'sgd':

                self.llama3_data[intent] = random.sample(self.llama3_data[intent], length)
                self.gpt_data[intent] = random.sample(self.gpt_data[intent] , length)
                self.llama3_refine_data[intent] = random.sample(self.llama3_refine_data[intent], length)
                self.llama3_supergen_data[intent] = random.sample(self.llama3_supergen_data[intent], length)
    
    
    @staticmethod
    def get_uni_bi_gram(doc):
        unigram = [token.text.lower() for token in doc if not token.is_punct]
        bigrams = [(doc[i].text, doc[i+1].text) for i in range(len(doc) - 1)]
        return unigram, bigrams
 
    
    def calculate_dist(self):
        nlp = spacy.load("en_core_web_sm")
        global_dist1 = [0] * 5 # because we have 5 file (real, gpt, llama3, llam3-refine, llama3-supergen) to comparison, each stands for the score
        global_dist2 = [0] * 5
        print(f'length:{len(self.real_data)},{len(self.llama3_data )},{len(self.gpt_data)},{len(self.llama3_refine_data)},{len(self.llama3_supergen_data)}')

        for intent in self.real_data:
            # print(f'length:{len(self.real_data[intent])}, {len(self.llama3_data[intent])},{len(self.gpt_data[intent])},{len(self.llama3_refine_data[intent])}')
            real_data = EvalDistinct.documentize(self.real_data)[intent]
            llama3_data = EvalDistinct.documentize(self.llama3_data)[intent]
            gpt_data = EvalDistinct.documentize(self.gpt_data)[intent]
            llama3_refine_data = EvalDistinct.documentize(self.llama3_refine_data)[intent]
            llama3_supergen_data = EvalDistinct.documentize(self.llama3_supergen_data)[intent]
            
            doc_by_datasource = [nlp(real_data),nlp(llama3_data),nlp(gpt_data), nlp(llama3_refine_data),nlp(llama3_supergen_data)]
            unigram_by_datasource = []
            bigram_by_datasource = []
            
            for doc in doc_by_datasource:
                unigram, bigrams = EvalDistinct.get_uni_bi_gram(doc)
                unigram_by_datasource.append(unigram)
                bigram_by_datasource.append(bigrams)
            
            min_unigram_len = float('inf')
            min_bigram_len = float('inf')
            
            for uni, bi in zip(unigram_by_datasource,bigram_by_datasource):
                if len(uni) < min_unigram_len:
                    min_unigram_len = len(uni) 
                if len(bi) < min_bigram_len:
                    min_bigram_len = len(bi)
                    
            unigram_by_datasource = [random.sample(doc, min_unigram_len) for doc in unigram_by_datasource]
            bigram_by_datasource = [random.sample(doc, min_bigram_len) for doc in bigram_by_datasource]
            
            dist1 = [len(set(doc))/len(doc) for doc in unigram_by_datasource]
            dist2 = [len(set(doc))/len(doc) for doc in bigram_by_datasource]

            global_dist1 = [x + y for x, y in zip(global_dist1, dist1)]
            global_dist2 = [x + y for x, y in zip(global_dist2, dist2)]
            
            
        global_dist1 = [dist1/len(self.llama3_data) for dist1 in global_dist1]
        global_dist2 = [dist2/len(self.llama3_data) for dist2 in global_dist2]

        return global_dist1, global_dist2
        
        
    

def log_dist(run_name, dist1, dist2, csv_file='../evaluation_record/dist_n_FT.csv'):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
                # Write header if the file does not exist
                writer.writerow(['run_name', 'dist-1', 'dist2'])
        for name, d1, d2 in zip(run_name, dist1, dist2):
            writer.writerow([name, d1, d2])
    




parser = argparse.ArgumentParser(description='To evaluate the refined data')
parser.add_argument('--dataset_name', default = 'sgd',type=str)
parser.add_argument('--domain_map_filepath', type=str)


args = parser.parse_args()
dataset_name = args.dataset_name
domain_map_filepath = args.domain_map_filepath

if dataset_name == 'sgd':
    real_data_filepath = '../data/sgd/train_dev_test.csv'

elif dataset_name == 'clinc150':
    real_data_filepath = '../data/all_rd_train.csv'




random.seed(42)



global_dist1 = [0]*5 # because we have 5 file (real, gpt, llama3, llam3-refine, llama3-supergen) to comparison, each stands for the score
global_dist2 = [0]*5
data_filename = [f'{dataset_name}_real', f'{dataset_name}_llama3', f'{dataset_name}_gpt', 
                 f'{dataset_name}_llama3_refine', f'{dataset_name}_llama3_supergen']

for seed_value in range(1,6):
    
    
    # load model_val for the labels
    ckpt_wd_val = f'../valid_classifer/distilbert-base-uncased_{dataset_name}_{seed_value}sd'
    pattern = re.compile(r'^checkpoint-\d+$')
    existing_ckpt = [d for d in os.listdir(ckpt_wd_val) if os.path.isdir(os.path.join(ckpt_wd_val, d)) and pattern.match(d)]

    checkpoint_name = existing_ckpt[0]
    ckpt_path_val = os.path.join(ckpt_wd_val, checkpoint_name)
    print(f'model_val checkpoint path: {ckpt_path_val}')
    model_val = AutoModelForSequenceClassification.from_pretrained(ckpt_path_val)
    
    
    
    gpt_data_filepath = f'../data/{dataset_name}/diversity/diversity_chatgpt_{dataset_name}_{seed_value}sd.json'
    llama3_data_filepath = f'../data/{dataset_name}/llama3_{dataset_name}_{seed_value}sd.csv'
    llama3_refine_data_filepath = f'../data/{dataset_name}/diversity/diversity_llama3_{dataset_name}_FT_ut72ut1_{seed_value}sd.json'
    llama3_supergen_data_filepath = f'../data/{dataset_name}/diversity/diversity_supergen_llama3_{dataset_name}_{seed_value}sd.json'

    eval_dist = EvalDistinct( dataset_name =  dataset_name, model_val = model_val)
    eval_dist.load_test_domain_intent_map(domain_map_filepath = domain_map_filepath)
    eval_dist.load_all_data(real_data_filepath = real_data_filepath,
                            gpt_data_filepath = gpt_data_filepath, 
                            llama3_data_filepath = llama3_data_filepath,
                            llama3_refine_data_filepath =llama3_refine_data_filepath,
                            llama3_supergen_data_filepath = llama3_supergen_data_filepath)
    dist1, dist2 = eval_dist.calculate_dist()
    print(f'{seed_value}sd result: dist1: {len(dist1)}; dist2: {dist2}')
    global_dist1 = [x + y for x, y in zip(global_dist1, dist1)]
    global_dist2 = [x + y for x, y in zip(global_dist2, dist2)]
    
global_dist1 = [round(x/5,3) for x in global_dist1]
global_dist2 = [round(x/5,3) for x in global_dist2]


log_dist(run_name = data_filename, 
         dist1 = global_dist1, 
         dist2 = global_dist2)