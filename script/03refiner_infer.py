
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, set_seed, AutoModelForSequenceClassification
import numpy as np
import re
import csv
import torch
import argparse
import random
from torch import cuda
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import Dataset
import os
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import time
class InferRefiner:
    def __init__(self, model, tokenizer, model_val,
                 dataset_name,input_ut, output_ut):

        self.model = model
        self.tokenizer = tokenizer
        self.model_val = model_val
        self.dataset_name = dataset_name          
        self.input_ut = input_ut
        self.output_ut = output_ut   

        self.domain_map_test = None
        self.intent_map_test = None
        
        self.list_of_uts = None
        self.list_of_intent = None
        self.test_data = dict() # test data in the script refers to llm generated unseen data
        self.list_of_refined_uts = []
        self.refined_intents_uts = []
    
    def load_llm_gen_data(self,llm_data_filepath):
        list_of_uts = []
        list_of_intent = []

        with open(llm_data_filepath, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            
            # Read each row from the CSV file and append it to the list_of_lists
            for row in csvreader:
                ut = row[0]
                intent = row[1]
                list_of_uts.append(ut)
                list_of_intent.append(intent)
        
        self.list_of_uts = list_of_uts
        self.list_of_intent = list_of_intent 
        print(f'All row data: length of uts and intents are {len(self.list_of_uts)} and {len(self.list_of_intent)}')      
        
    def load_test_domain_intent_map(self, domain_map_filepath, test_split = True):
        
        with open(domain_map_filepath, 'r') as file:
            # Load the JSON data from the file
            domain_map = json.load(file)
        intent_list_train = list(self.model_val.config.label2id.keys())
        intent_map_test = dict()
        domain_map_test = dict()
        
        if test_split:
            for dm in domain_map:
                intent_list = domain_map[dm]
                for intent in intent_list:
                    if intent not in intent_list_train:
                        intent_map_test[intent] = dm
                        if dm not in domain_map_test:
                            domain_map_test[dm] = [intent]
                        else:
                            domain_map_test[dm].append(intent)
        else:
            for dm in domain_map:
                intent_list = domain_map[dm]
                for intent in intent_list:
                    if intent in intent_list_train:
                        intent_map_test[intent] = dm
                        if dm not in domain_map_test:
                            domain_map_test[dm] = [intent]
                        else:
                            domain_map_test[dm].append(intent)
                     
                
            
        
        print(f'Test domain is : {list(domain_map_test.keys())}')
        print(f'Total test intent number : {len(intent_map_test.keys())}')        
        

        self.domain_map_test = domain_map_test
        self.intent_map_test = intent_map_test

    
    
    def get_test_data_dic(self,llm_data_filepath,domain_map_filepath,test_split = True):
        self.load_test_domain_intent_map(domain_map_filepath = domain_map_filepath, test_split = test_split)
        test_intent_list = list(self.intent_map_test.keys())
        for intent, ut in zip(self.list_of_intent, self.list_of_uts):
            if intent in test_intent_list:
                if intent not in self.test_data:
                    self.test_data[intent] = [ut]
                else:
                    self.test_data[intent].append(ut)

    def adjust_intents(self, input_string):
        """
        Becasue in the sgd dataset, the intent is like this form: BookTicket. We have to make it seperate and into lower case.
        """
        if self.dataset_name == 'clinc150':
            output_string = input_string.replace('_', ' ')
        elif self.dataset_name == 'sgd':
            words = re.findall('[A-Z][^A-Z]*', input_string)
            if not words:
                raise ValueError("No words found in the adjust-intent-list")
            output_string = ' '.join(words).lower()
        return output_string   

    def data_processing(self,llm_data_filepath,domain_map_filepath, test_split = True):
        self.get_test_data_dic(llm_data_filepath = llm_data_filepath,
                               domain_map_filepath =domain_map_filepath,
                               test_split = test_split)
        # clear the data
        self.list_of_uts = [] 
        self.list_of_intent = []

        for intent, ut in self.test_data.items():
            if self.dataset_name == 'sgd':
                gen_uts = ut[:200]
            elif self.dataset_name == 'clinc150':
                gen_uts = ut[:100]
            else:
                raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
            
            tmp_intent = [intent] * len(gen_uts)    
            self.list_of_intent += tmp_intent
            intent_text = self.adjust_intents(intent)
            single_dm = self.intent_map_test[intent]

            
            if self.input_ut > 1:   
                if self.output_ut == 1:
                    prefix_wd = f"\nThe sentences above are user intent expressions for \"{intent_text}\" in the \"{single_dm}\" context, but they might have less quality or contain mistakes. Provide one improved expression.\n"
                elif self.output_ut >1:
                    prefix_wd = f"\nThe sentences above are user intent expressions for \"{intent_text}\" in the \"{single_dm}\" context, but they might have less quality or contain mistakes. Provide {self.output_ut} improved expressions.\n"
                
                gen_uts_chunk = [
                    '\n'.join([gen_uts[i]] + random.sample([x for x in gen_uts if x != gen_uts[i]], k=self.input_ut - 1)) + prefix_wd
                    for i in range(len(gen_uts))
                ]                
            else:
                prefix_wd = f"\nThe sentence above is a user intent expression for \"{intent_text}\" in the \"{single_dm}\" context, but they might have less quality or contain mistakes. Provide one improved expression.\n"
                gen_uts_chunk = [a_ut + prefix_wd for a_ut in gen_uts]            
            self.list_of_uts += gen_uts_chunk


        print(f'All llm-generated-unseen data: length of uts and intents are {len(self.list_of_uts)} and {len(self.list_of_intent)}')      
        print(f'One example from llm-generated-unseen data [list_of_uts]:\n{self.list_of_uts[0]} ')
        print(f'One example from llm-generated-unseen data [list_of_intent]:\n{self.list_of_intent[0]}')
            
    def preprocess_function(self,examples): #13
        max_input_length = 512
        model_inputs = self.tokenizer(
            examples['input_uts'],
            max_length=max_input_length,
            truncation=True,
        )
        return model_inputs            

    def split_into_one_by_one_pairs(self):
        """
        Because the data output can be multiple uts, we have to split into one to pair its intent
        """
        result = []
        count = 0
        tot_len = 0
        for row in self.refined_intents_uts:
            uts = row[0]
            intent = row[1]
            list_of_uts = uts.split('<sep>')
            list_of_exps = [[ut.strip(),intent] for ut in list_of_uts]
            result += list_of_exps
            
            count+= 1
            tot_len += len(list_of_exps)        
        print(f'The average output ut num is {tot_len/count}')
        self.refined_intents_uts = result

    def save_to_csv(self, filepath):

        with open(filepath, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(self.refined_intents_uts)
        print(f'Data has been saved to {filepath}')
            
    
    def inference_batch(self, num_of_gen, filepath):
        data_dict_list = {"input_uts": [exp for exp in self.list_of_uts]} 

        # Create a Dataset object
        val_dataset = Dataset.from_dict(data_dict_list)


        tokenized_datasets = val_dataset.map(self.preprocess_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['input_uts'])
        print(f'tokenized_datasets: {tokenized_datasets}')
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=self.model)
        val_params = {'batch_size': 100,'shuffle': False,'num_workers': 0,'collate_fn':data_collator}
        val_loader = DataLoader(tokenized_datasets, **val_params) 
        max_new_tokens = 70 * self.output_ut
        print(f'max_new_tokens: 70 * {self.output_ut}')
        start_time = time.time()
        verbose = 0 
        for i in range(num_of_gen):
            for batch in  tqdm(val_loader):
                verbose += 1
                batch = {k: v.to(device) for k, v in batch.items()}
                output_tokens = self.model.generate(**batch, do_sample = True, max_new_tokens=max_new_tokens)
                predictions = [self.tokenizer.decode(output_token, skip_special_tokens=True) for output_token in output_tokens]
                self.list_of_refined_uts += predictions
                
                dmo = predictions[0]
                print(f'One prediction example:\n{dmo}')
            print(f'The length of pedictions: {len(self.list_of_refined_uts)} , and lables: {len(self.list_of_intent)}')
            self.refined_intents_uts += [[ut,intent] for ut, intent in zip(self.list_of_refined_uts,self.list_of_intent)]
            self.list_of_refined_uts = []

        self.split_into_one_by_one_pairs()
        self.save_to_csv(filepath = filepath)
            

        end_time = time.time()

        # Calculate the total time spent
        total_time = end_time - start_time
        print(f"Total time spent: {total_time:.2f} seconds")    






def extract_number_from_path(path,dataset_name):
    # Debug print to check the input path
    print(f"Input path: {path}")
    
    # Use regex to search for the pattern with alternation for clinc150 and sgd
    match = re.search(r'_(clinc150|sgd)_(\d+)sd', path)
    
    # Check if a match was found
    if match:
        # Debug print to show the matched part
        return int(match.group(2))  # Use group 2 for the digits
    else:
        raise ValueError("sd value not found in the filename")


def find_checkpoint(wd):
    """
    given the work directory where the checkpoint folder belong, it will return full path include the checkpoint
    """
    pattern = re.compile(r'^checkpoint-\d+$')
    existing_ckpt = [d for d in os.listdir(wd) if os.path.isdir(os.path.join(wd, d)) and pattern.match(d)]

    checkpoint_name = existing_ckpt[0]
    ckpt_path = os.path.join(wd, checkpoint_name)
    return ckpt_path



parser = argparse.ArgumentParser(description='data refiner infer')
parser.add_argument('--llm_data_filepath', help='The csv file from 01prompting LLM generation')
parser.add_argument('--model_wd', help='The refiner path')
parser.add_argument('--domain_map_filepath', type=str)



args = parser.parse_args()

llm_data_filepath =args.llm_data_filepath
model_wd = args.model_wd
domain_map_filepath = args.domain_map_filepath


if 'sgd' in llm_data_filepath:
    dataset_name = 'sgd'
    
elif 'clinc150' in llm_data_filepath:
    dataset_name = 'clinc150'


if 'llama3' in llm_data_filepath:
    llm = 'llama3'
elif 'zphr' in llm_data_filepath:
    llm = 'zphr'
else:
    print("No match found")
    raise ValueError("llm value is missing")


device = 'cuda' if cuda.is_available() else 'cpu'
seed_value = extract_number_from_path(path = llm_data_filepath, dataset_name = dataset_name)

 



if 'ut' not in model_wd:
    model_ck = model_wd
    input_ut = 7
    output_ut = 1
    save_filepath = f'../data/{dataset_name}/Nf/{llm}_{dataset_name}_ut{input_ut}2ut{output_ut}_nf_{seed_value}sd.csv'

else:
    model_ck = find_checkpoint(model_wd)
    last_slash_index = model_wd.rfind('/')
    input_output = model_wd.split('_')[-1]
    input_ut = int(input_output[2])
    output_ut = int(input_output[-1])
    if 'Peft' in model_wd:
        save_filepath = f'../data/{dataset_name}/refine/{llm}_{dataset_name}_ut{input_ut}2ut{output_ut}_{seed_value}sd.csv'
    else:
        save_filepath = f'../data/{dataset_name}/refine/{llm}_{dataset_name}_FT_ut{input_ut}2ut{output_ut}_{seed_value}sd.csv'


print(f'Generater model full checkpoint path: {model_ck}')
print(f'save_filepath: {save_filepath}')
if input_ut == 7:
    num_of_gen = 3
else:
    num_of_gen = 1


# Reproducibility
torch.manual_seed(seed_value)
np.random.seed(seed_value) 
torch.backends.cudnn.deterministic = True
set_seed(seed_value)
random.seed(seed_value) 


# Load refiner

if "Peft" in model_ck:

    peft_config = PeftConfig.from_pretrained(model_ck)
    model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, model_ck)
    model = model.to(device)

else: 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ck)
    model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_ck)


# load model_val for the labels
ckpt_wd_val = f'../valid_classifer/distilbert-base-uncased_{dataset_name}_{seed_value}sd'
pattern = re.compile(r'^checkpoint-\d+$')
existing_ckpt = [d for d in os.listdir(ckpt_wd_val) if os.path.isdir(os.path.join(ckpt_wd_val, d)) and pattern.match(d)]

checkpoint_name = existing_ckpt[0]
ckpt_path_val = os.path.join(ckpt_wd_val, checkpoint_name)
print(f'model_val checkpoint path: {ckpt_path_val}')


model_val = AutoModelForSequenceClassification.from_pretrained(ckpt_path_val)
model_val = model_val.to(device)
tokenizer_val = AutoTokenizer.from_pretrained(ckpt_path_val)






refiner_infer = InferRefiner(model = model, tokenizer = tokenizer, model_val = model_val,
                             dataset_name = dataset_name,input_ut = input_ut, output_ut = output_ut)
refiner_infer.load_llm_gen_data(llm_data_filepath = llm_data_filepath)
refiner_infer.data_processing(llm_data_filepath = llm_data_filepath,domain_map_filepath = domain_map_filepath, test_split = True)
refiner_infer.inference_batch(num_of_gen = num_of_gen, filepath =  save_filepath)