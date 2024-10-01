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

class EvalRefiner:
    def __init__(self,  dataset_name, model_val, 
                 device, seed):
        

        self.model_val = model_val
        self.device = device
        self.dataset_name = dataset_name
          
        self.seed = seed

        self.llm_data =  dict()
        self.real_test_data =  dict()
        self.real_data = None # this one is used for dist-n check
        
        self.list_of_uts = None
        self.list_of_intent = None         
        self.list_of_uts_test = None
        self.list_of_intent_test = None   
        self.train_data = None   # this one will be used for train classfier and eval     
        self.domain_map_test = None
        self.intent_map_test = None 
        self.accuracy = None


    @staticmethod
    def load_csv_data(filepath):
        list_of_uts = []
        list_of_intent = []

        with open(filepath, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            
            # Read each row from the CSV file and append it to the list_of_lists
            for row in csvreader:
                ut = row[0]
                intent = row[1]
                list_of_uts.append(ut)
                list_of_intent.append(intent)
        return list_of_uts, list_of_intent
    
    def load_llm_and_real_data(self,llm_data_filepath, real_test_data_filepath):
        self.list_of_uts, self.list_of_intent = EvalRefiner.load_csv_data(llm_data_filepath)
        self.list_of_uts_test, self.list_of_intent_test = EvalRefiner.load_csv_data(real_test_data_filepath)

 
    

    def load_test_domain_intent_map(self, domain_map_filepath):
        with open(domain_map_filepath, 'r') as file:
            # Load the JSON data from the file
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
    def data_to_dic(list_of_uts, list_of_intent, test_intent_list):
        data_dic = dict()
        for intent, ut in zip(list_of_intent, list_of_uts):
            if intent in test_intent_list:
                if intent not in data_dic:
                    data_dic[intent] = [ut]
                else:
                    data_dic[intent].append(ut)
        return  data_dic
        

        
    def keep_unseen_intent_and_trunk_samplesize(self, mutiplier = 1):
        self.load_test_domain_intent_map(domain_map_filepath = domain_map_filepath)
        test_intent_list = list(self.intent_map_test.keys())

        self.llm_data = EvalRefiner.data_to_dic(list_of_uts = self.list_of_uts, 
                                                list_of_intent = self.list_of_intent, 
                                                test_intent_list = test_intent_list)
        self.real_test_data = EvalRefiner.data_to_dic(list_of_uts = self.list_of_uts_test, 
                                                      list_of_intent = self.list_of_intent_test, 
                                                      test_intent_list = test_intent_list)

        if self.dataset_name == 'sgd':
            train_data = {intent: ut_list if len(ut_list) <= 200 * mutiplier else random.sample(ut_list, int(200 * mutiplier)) for intent, ut_list in self.llm_data.items()}
            # train_data = {intent: ut_list[:200] for intent, ut_list in self.llm_data.items()}
        elif self.dataset_name == 'clinc150':
            train_data = {intent: ut_list if len(ut_list) <= 100 * mutiplier else random.sample(ut_list, int(100 * mutiplier)) for intent, ut_list in self.llm_data.items()}
            # train_data = {intent: ut_list[:100] for intent, ut_list in self.llm_data.items()}
        else:
            raise ValueError("dataset_name value not qualified") 
        
        self.train_data = train_data
    def preprocess_function(self, examples):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return self.tokenizer(examples["text"], truncation=True)
    
    def compute_metrics_eval(self,eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = evaluate.load("accuracy")
        return accuracy.compute(predictions=predictions, references=labels)
    
    def load_or_train_model(self, data_filename):
        save_wd = f"../eval_classifer/distilbert-base-uncased_{data_filename}"
        print(f'The model will be saved in the path: {save_wd}')
        if not os.path.exists(save_wd):
            print('Did not found the eval classfier, train a new one:')
            train_data = self.train_data
            test_intent_list = list(self.intent_map_test.keys())
            id2label = {num:label for num,label in enumerate(test_intent_list)}
            label2id = {label:num for num,label in enumerate(test_intent_list)}
            data_dic = {'label':[], 'text':[]}
            for intent, ut_list in train_data.items():
                for ut in ut_list:
                    label_id = label2id[intent]
                    data_dic ['text'].append(ut)
                    data_dic ['label'].append(label_id)
            train_data = Dataset.from_dict(data_dic)
            train_data = train_data.shuffle(seed=self.seed)
            train_data = train_data.train_test_split(test_size=0.2, seed=self.seed)
            train_tkd = train_data.map(self.preprocess_function, batched=True)
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            train_step_num = 1800
            print(f"the  train_tkd: {train_tkd}")
            step_num = 20
            self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id)
            self.model = self.model.to(self.device)

             
            training_args = TrainingArguments(
                output_dir= save_wd,
                learning_rate=2e-5,
                per_device_train_batch_size=60, 
                per_device_eval_batch_size=60, 
                max_steps = train_step_num, # previously use 1500, 600
                weight_decay=0.01,
                seed = self.seed,
                evaluation_strategy="steps",
                eval_steps = step_num,
                save_strategy='steps',
                save_steps = step_num,
                save_total_limit = 1,
                logging_strategy = 'steps',
                logging_steps = step_num,
                load_best_model_at_end=True,
                metric_for_best_model='accuracy',
                report_to=None,
                push_to_hub=False,
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_tkd['train'],
                eval_dataset=train_tkd['test'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics_eval,
                callbacks=[EarlyStoppingCallback(early_stopping_patience = 5)]
            )

            trainer.train()




        pattern = re.compile(r'^checkpoint-\d+$')
        existing_ckpt = [d for d in os.listdir(save_wd) if os.path.isdir(os.path.join(save_wd, d)) and pattern.match(d)]
        if  os.path.exists(save_wd):
            print(f'existing_ckpt found, load it: {existing_ckpt}')

        checkpoint_name = existing_ckpt[0]
        ckpt_path = os.path.join(save_wd, checkpoint_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    def eval_accuracy(self):
        intent_list = []
        ut_list = []
        for intent, ut_lists in self.real_test_data.items():
            for ut in ut_lists:
                ut_list.append(ut)
                intent_list.append(intent)
        print(f'There are {len(intent_list)} testing examples.')        
        test_dataset = Dataset.from_dict({"text": ut_list})
        tokenized_dataset = test_dataset.map(self.preprocess_function, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        data_collator = DataCollatorWithPadding(self.tokenizer)
        batch_size = 100
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)


        predicted_intent_list = []

        # Inference loop
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.model.device) for k, v in batch.items()}
                logits = self.model(**inputs).logits
                predicted_class_ids = logits.argmax(dim=1).tolist()
                
                for predicted_class_id in predicted_class_ids:
                    predicted_label = self.model.config.id2label[predicted_class_id]
                    predicted_intent_list.append(predicted_label)
        print(f'length of gd and pred are the same : {len(intent_list) == len(predicted_intent_list)}')
        accuracy = accuracy_score(intent_list, predicted_intent_list)
        self.accuracy = round(float(accuracy)*100,1)
        print(f"Accuracy: {self.accuracy}")
        


    def log_accuracy(self, run_name, csv_file='../evaluation_record/accuracy_log.csv'):
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write header if the file does not exist
                writer.writerow(['run_name', 'accuracy'])
            # Write the run_name and accuracy
            writer.writerow([run_name, self.accuracy])




    def get_eval_data_for_dist(self):
        filepath = f"../data/{self.dataset_name}/diversity/diversity_{data_filename}.json"
        with open(filepath, 'w') as json_file:
            json.dump(self.train_data, json_file)
        print(f'data has been saved in the path: {filepath}')

  




def extract_seed_number(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split("_")
    for part in parts:
        if part.endswith("sd.csv"):
            seed_number = part[:-len("sd.csv")]
            if seed_number.isdigit():
                return int(seed_number)
    return ValueError("sd value not found in the filename")





parser = argparse.ArgumentParser(description='To evaluate the refined data')
parser.add_argument('--llm_data_filepath', default = '../data/sgd/chatgpt_sgd_2sd.csv',type=str)
parser.add_argument('--log_accuracy', default = 'yes',type=str)
parser.add_argument('--mutiplier', default = 1,type=float)
parser.add_argument('--domain_map_filepath', type=str)

args = parser.parse_args()
llm_data_filepath = args.llm_data_filepath
log_accuracy = args.log_accuracy
mutiplier = args.mutiplier
domain_map_filepath = args.domain_map_filepath

os.environ["WANDB_DISABLED"] = "true"


if 'sgd' in llm_data_filepath:
    dataset_name = 'sgd'
    real_test_data_filepath = '../data/sgd/train_dev_test.csv'
    real_train_data_filepath = '../data/sgd/train_dev_test.csv'
    
elif 'clinc150' in llm_data_filepath:
    dataset_name = 'clinc150'
    real_test_data_filepath = '../data/all_rd_test.csv'
    real_train_data_filepath = '../data/all_rd_train.csv'
 
data_filename = os.path.splitext(os.path.basename(llm_data_filepath))[0]  
if mutiplier != 1:
    data_filename = f'{mutiplier}x_' + data_filename
print(f'data filename is: {data_filename}')

device = 'cuda' if cuda.is_available() else 'cpu'
seed_value = extract_seed_number(filepath = llm_data_filepath)    
print(f'Seed value is: {seed_value}')
# Reproducibility
torch.manual_seed(seed_value)
np.random.seed(seed_value) 
torch.backends.cudnn.deterministic = True
set_seed(seed_value)
random.seed(seed_value) 





# load model_val for the labels
ckpt_wd_val = f'../valid_classifer/distilbert-base-uncased_{dataset_name}_{seed_value}sd'
pattern = re.compile(r'^checkpoint-\d+$')
existing_ckpt = [d for d in os.listdir(ckpt_wd_val) if os.path.isdir(os.path.join(ckpt_wd_val, d)) and pattern.match(d)]

checkpoint_name = existing_ckpt[0]
ckpt_path_val = os.path.join(ckpt_wd_val, checkpoint_name)
print(f'model_val checkpoint path: {ckpt_path_val}')
model_val = AutoModelForSequenceClassification.from_pretrained(ckpt_path_val)
model_val = model_val.to(device)

eval_refiner = EvalRefiner(dataset_name = dataset_name, model_val = model_val, device = device, seed =seed_value)

eval_refiner.load_llm_and_real_data(llm_data_filepath = llm_data_filepath,
                           real_test_data_filepath = real_test_data_filepath)

eval_refiner.keep_unseen_intent_and_trunk_samplesize(mutiplier = mutiplier)


if log_accuracy == 'yes':
    eval_refiner.load_or_train_model(data_filename = data_filename)
    eval_refiner.eval_accuracy()
    eval_refiner.log_accuracy(run_name = data_filename)

            