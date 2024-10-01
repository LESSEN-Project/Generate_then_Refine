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

class TrainRefiner:
    
    def __init__(self,  dataset_name, model, 
                 tokenizer, device,input_ut, 
                 output_ut, seed):
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_name = dataset_name
        self.input_ut = input_ut
        self.output_ut = output_ut

        self.model_val = None
        self.tokenizer_val = None            
        self.seed = seed
        self.domain_map = None
        self.intent_map = None
        self.domain_split = None
        self.llm_data = None
        self.real_data = None
        self.train_data = None
        self.val_data = None


        self.best_val_loss = float('inf')  # Initialize the best validation loss to infinity
        self.previous_ckpt_path = None  # To keep track of the previous checkpoint
        self.patience = 0
    
    @staticmethod
    def load_csv_file(filepath):
        """
        Returns:
        - a dict: key is intent and value is a list of its utterance
        """
        examples = []
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                examples.append(row)
                
        intent_uts_dict = dict()
        for exp in examples:
            ut = exp[0]
            intent = exp[1]
            if intent not in intent_uts_dict:
                intent_uts_dict[intent] = [ut]
            else:
                intent_uts_dict[intent].append(ut)
        return  intent_uts_dict
    def load_all_data(self,llm_data_filepath, real_data_filepath, domain_map_filepath):
        """
        The method will load llm data, real data, domain map and intent map
        """
        llm_data = TrainRefiner.load_csv_file(llm_data_filepath)
        real_data = TrainRefiner.load_csv_file(real_data_filepath)
        
        with open(domain_map_filepath, 'r') as file:
            # Load the JSON data from the file
            domain_map = json.load(file)
        
        intent_map = dict()
        for dm in domain_map:
            intent_list = domain_map[dm]
            for intent in intent_list:
                intent_map[intent] = dm
        
        self.llm_data = llm_data
        self.real_data = real_data
        self.domain_map = domain_map
        self.intent_map = intent_map
    

    @staticmethod
    def get_unselected_domains(original_dms, selected_dms):
        unselected_dms = []
        for dm in original_dms:
            if dm not in selected_dms:
                unselected_dms.append(dm)
        return unselected_dms
    
    def get_train_valid_eval_domain_split(self):
        """
        The filepath is the json file of a domain map (key:value - domain:intent_list)
        Output:
        it it will be a dict of domain splits{train:[], val:[], test:[]}
        """


        
        domains = list(self.domain_map.keys())
        
        
        
        if self.dataset_name == 'clinc150':
            half_size = len(domains) // 2
            test_split = random.sample(domains, half_size)
            num_valid_dm = 1
            
        elif self.dataset_name == 'sgd':
            half_size = len(domains) // 2
            test_split = random.sample(domains, half_size - 2)
            num_valid_dm = 3
            
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
            
        train_valid_split = TrainRefiner.get_unselected_domains(original_dms = domains, selected_dms = test_split)
        valid_split = random.sample(train_valid_split, num_valid_dm)
        train_split = TrainRefiner.get_unselected_domains(original_dms = train_valid_split, selected_dms = valid_split)
        self.domain_split = {'train': train_split, 'val': valid_split, 'test': test_split }
        print(f'The domain split is: {self.domain_split}')
                


   
    def load_domain_label(self, domains):
        result = []
        for dm in domains:
            result += self.domain_map[dm]
        return result
    
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
    
    
    def preprocess_function(self, examples): #13
        max_input_length = 512
        max_target_length = 512
        model_inputs = self.tokenizer(
            examples["ut_list"],
            max_length=max_input_length,
            truncation=True,
        )
        
        labels = self.tokenizer(examples["ut_list_rd"], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def preprocess_function_model_val(self, examples):
        self.tokenizer_val = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return self.tokenizer_val(examples["text"], truncation=True)
    
    def compute_metrics_eval(self,eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = evaluate.load("accuracy")
        return accuracy.compute(predictions=predictions, references=labels)
    
    def train_model_val(self):
        save_wd = f"../valid_classifer/distilbert-base-uncased_{self.dataset_name}_{self.seed}sd"
        if not os.path.exists(save_wd):
            
            intent_val = self.load_domain_label(domains = self.domain_split['val'])
            intent_train = self.load_domain_label(domains = self.domain_split['train'])
            intent_list_tmp  = intent_val + intent_train
            intent_list = []
            for intent in intent_list_tmp:
                if intent not in intent_list:
                    intent_list.append(intent)
            print(f'There are {len(intent_list)} seen intents.')
            train_data = {intent: ut_list[:200] for intent, ut_list in self.real_data.items() if intent in intent_list}

                    
            id2label = {num:label for num,label in enumerate(intent_list)}
            label2id = {label:num for num,label in enumerate(intent_list)}
            data_dic = {'label':[], 'text':[]}
            for intent, ut_list in train_data.items():
                for ut in ut_list:
                    label_id = label2id[intent]
                    data_dic ['text'].append(ut)
                    data_dic ['label'].append(label_id)
            train_data = Dataset.from_dict(data_dic)
            train_data = train_data.shuffle(seed=self.seed)
            train_data = train_data.train_test_split(test_size=0.2, seed=self.seed)
            train_tkd = train_data.map(self.preprocess_function_model_val, batched=True)
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer_val)

            train_step_num = 1800
            print(f"the length of training data: {len(train_tkd)}; taining steps num {train_step_num}")
            step_num = 20
            self.model_val = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id)
            self.model_val = self.model_val.to(self.device)

             
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
                model=self.model_val,
                args=training_args,
                train_dataset=train_tkd['train'],
                eval_dataset=train_tkd['test'],
                tokenizer=self.tokenizer_val,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics_eval,
                callbacks=[EarlyStoppingCallback(early_stopping_patience = 5)]
            )

            trainer.train()
            del trainer
            torch.cuda.empty_cache()



        pattern = re.compile(r'^checkpoint-\d+$')
        existing_ckpt = [d for d in os.listdir(save_wd) if os.path.isdir(os.path.join(save_wd, d)) and pattern.match(d)]
        print(f'existing_ckpt: {existing_ckpt}')

        checkpoint_name = existing_ckpt[0]
        ckpt_path = os.path.join(save_wd, checkpoint_name)
        self.model_val = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
        self.model_val = self.model_val.to(self.device)
        self.tokenizer_val = AutoTokenizer.from_pretrained(ckpt_path)



    
    def process_and_tokenize_loaded_data(self):
        
        gen_data = self.llm_data
        real_data = self.real_data
        
        # get the split of train_val_test
        intent_lst_train = []  # this is to record each ut's intent
        intent_lst_val = [] # this is to record each ut's intent
        train_dataset = {'ut_list':[], 'ut_list_rd':[], 'intnt_lst':None}
        val_dataset = {'ut_list':[], 'ut_list_rd':[], 'intnt_lst':None} 
        val_intent_list = self.load_domain_label(domains = self.domain_split['val']) # this is what intents in the domains where val belongs
        train_intent_list = self.load_domain_label(domains = self.domain_split['train']) # this is what intents in the domains where train belongs
        test_intent_list = self.load_domain_label(domains = self.domain_split['test']) # this is what intents in the domains where test belongs
        seen_intents = set(list(self.model_val.config.label2id.keys()))
        print(f'there are totoaling {len(seen_intents)} seen intents')
        
        for intent in gen_data:
            if intent not in seen_intents:
                continue
            else:

                real_uts = real_data[intent][:200]
                gen_uts = gen_data[intent][:len(real_uts)] 
                    
                intent_text = self.adjust_intents(intent)
                single_dm = self.intent_map[intent]
                # refiner input data process
                if self.input_ut > 1:   
                    if self.output_ut == 1:
                        prefix_wd = f"\nThe sentences above are user intent expressions for \"{intent_text}\" in the \"{single_dm}\" context, but they might have less quality or contain mistakes. Provide one improved expression.\n"
                    elif self.output_ut >1:
                        prefix_wd = f"\nThe sentences above are user intent expressions for \"{intent_text}\" in the \"{single_dm}\" context, but they might have less quality or contain mistakes. Provide {self.output_ut} improved expressions.\n"
                    
                    gen_uts_chunk = [
                        '\n'.join([gen_uts[i]] + random.sample([x for x in gen_uts if x != gen_uts[i]], k=self.input_ut - 1)) + prefix_wd
                        for i in range(len(gen_uts))
                    ]    
                                   
                    if intent not in val_intent_list:
                        train_dataset['ut_list']    += gen_uts_chunk
                        intent_lst_train+= [intent]*len(gen_uts) # for train loder
                    else:
                        val_dataset['ut_list']    += gen_uts_chunk
                        intent_lst_val+= [intent]*len(gen_uts) # for validation loder

                else:
                    prefix_wd = f"\nThe sentence above is a user intent expression for \"{intent_text}\" in the \"{single_dm}\" context, but they might have less quality or contain mistakes. Provide one improved expression.\n"
                    gen_uts_chunk = [a_ut + prefix_wd for a_ut in gen_uts]
                    if intent not in val_intent_list:
                        train_dataset['ut_list']    += gen_uts_chunk
                        intent_lst_train+= [intent]*len(gen_uts) # for train loder
                    else:
                        val_dataset['ut_list']+= gen_uts_chunk   
                        intent_lst_val+= [intent]*len(gen_uts) # for validation loder
            
                # refiner output data process
                if self.output_ut > 1:   

                    real_uts_chunk = [
                        '<sep>'.join([real_uts[i]] + random.sample([x for x in real_uts if x != real_uts[i]], k=self.output_ut - 1))
                        for i in range(len(real_uts))
                    ]
                    
                    
                    if intent not in val_intent_list:
                        train_dataset['ut_list_rd'] += real_uts_chunk
                    else:
                        val_dataset['ut_list_rd']    += real_uts_chunk  
                else:
                    if intent not in val_intent_list:
                        train_dataset['ut_list_rd'] += real_uts
                    else:
                        val_dataset['ut_list_rd'] += real_uts
               

        
        # refiner validation data process (keep the intents for the uts)
        intent_lst_train = [self.model_val.config.label2id[label] for label in intent_lst_train]
        intent_lst_val = [self.model_val.config.label2id[label] for label in intent_lst_val]
        train_dataset['intnt_lst'] = intent_lst_train
        val_dataset['intnt_lst'] = intent_lst_val

        len_val_rd =len(val_dataset['ut_list_rd'])
        len_val_intnt =len(val_dataset['intnt_lst'])
        print(f'len_val_rd is the same as len_val_intnt {len_val_rd} vs {len_val_intnt}')


        dmo_input  = train_dataset['ut_list'][0]
        dmo_output = train_dataset['ut_list_rd'][0]

        print(f'One exp for generator:')
        print(f'Input:\n{dmo_input}\nOutput:\n{dmo_output}')
            
        train_dataset = Dataset.from_dict(train_dataset)
        tk_train_datasets = train_dataset.map(self.preprocess_function, batched=True)
        tk_train_datasets = tk_train_datasets.remove_columns(['ut_list', 'ut_list_rd'])

        val_dataset = Dataset.from_dict(val_dataset)
        tk_val_datasets = val_dataset.map(self.preprocess_function, batched=True)
        tk_val_datasets = tk_val_datasets.remove_columns(['ut_list', 'ut_list_rd'])
        
        self.train_data = tk_train_datasets
        self.val_data = tk_val_datasets

    def get_cl_input(self,batch, count_verbo):
        self.model.eval()
        intnt_labels = batch['intnt_lst']
        if count_verbo == 0: print(f'orign len of intent labels: {len(intnt_labels)}')
        batch = {k: v.to(self.device) for k, v in batch.items() if k == 'input_ids' or k == 'attention_mask'}
        with torch.no_grad():
            batch_gen = self.model.generate(**batch, do_sample = True, min_new_tokens = 3, max_new_tokens=70 * self.output_ut)
        predictions = [self.tokenizer.decode(output_token, skip_special_tokens=True) for output_token in batch_gen]
        
        if self.output_ut > 1:
            predictions_muti_output = []
            intent_muti_output = []
            for intnt, uts in zip(intnt_labels,predictions):
                lst_of_uts = uts.split('<sep>')
                predictions_muti_output += lst_of_uts
                
                intnt_tmp = [intnt] * len(lst_of_uts)
                intent_muti_output += intnt_tmp
            
            predictions = predictions_muti_output
            intnt_labels = intent_muti_output
            intnt_labels = torch.tensor(intnt_labels, dtype=torch.int64)
        # for visualization -start
        mid_ind = int(len(intnt_labels)//2)
        demo0,demo1,demo2 = predictions[0],predictions[mid_ind], predictions[-1]
        intent_ids = intnt_labels[0],intnt_labels[mid_ind], intnt_labels[-1]
        intent_lbs = [self.model_val.config.id2label[id.item()] for id in intent_ids]
        if count_verbo % 30 == 0:
            print(f'some prediction:')
            print(f'len of the prediction is:{len(predictions)}')
            print(f'Intent: {intent_lbs[0]}; Prediction: {demo0}')
            print(f'Intent: {intent_lbs[1]}; Prediction: {demo1}')
            print(f'Intent: {intent_lbs[2]}; Prediction: {demo2}')
        
        if count_verbo == 0: 
            print(f'adjusted len of intent labels: {len(intnt_labels)}')
            print(f'len of predictions: {len(predictions)}')
            print(f'len of intnt_labels: {len(intnt_labels)}')
        # for visualization -end
            

        tk_pred_val = self.tokenizer_val(predictions, padding=True, truncation=True, return_tensors='pt')
        input_ids, attention_mask = tk_pred_val['input_ids'], tk_pred_val['attention_mask']
        return input_ids, attention_mask, intnt_labels
    
    
    def validate_gen(self, loader):
        print(f'Validating generator.... The length of validation loader is {len(loader)}')
        self.model_val.eval()
        self.model.eval()
        with torch.no_grad():
            val_loss1 = 0
            val_loss3 = 0
            count_verbo = 0
            for _, batch in enumerate(loader, 0):  
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'], 
                                labels=batch['labels'],
                                decoder_input_ids = batch['decoder_input_ids'])    
                loss1 = outputs.loss
                loss1 = torch.mean(loss1)       
                val_loss1 += loss1.item()
                
                
                input_ids, attention_mask, intnt_labels = self.get_cl_input(batch = batch, count_verbo = count_verbo)
                input_ids, attention_mask, intnt_labels = input_ids.to(self.device), attention_mask.to(self.device), intnt_labels.to(self.device)
                outputs_val =  self.model_val(input_ids=input_ids, 
                                        attention_mask=attention_mask, 
                                        labels=intnt_labels)
                count_verbo += 1
                if output_ut > 1:
                    batch_size = next(iter(batch.values())).size(0)
                    same_size_normalizaer = len(intnt_labels)/batch_size
                    loss3 = outputs_val.loss / same_size_normalizaer
                else:
                    loss3 = outputs_val.loss
                loss3 = torch.mean(loss3)
                val_loss3 += loss3.item()
            val_loss1_avg =  val_loss1/len(loader)    
            val_loss3_avg =  val_loss3/len(loader)

                
        return val_loss1_avg, val_loss3_avg
    def modeling(self, grc_steps, epoch_num, lr, batch_size = 24):
        
        # put data in data loader
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model= self.model) 
        train_params = {'batch_size': int(batch_size/grc_steps),'shuffle': True,'num_workers': 0,'collate_fn':data_collator}
        val_params = {'batch_size': batch_size,'shuffle': False,'num_workers': 0,'collate_fn':data_collator}

        training_loader = DataLoader(self.train_data, **train_params) 
        val_loader = DataLoader(self.val_data, **val_params)     
        
        # Training for loop

        optimizer = torch.optim.Adam(params =  self.model.parameters(), lr=lr)
        print(f"training_loader {len(training_loader)}")


        start_time = time.time()

        step = 0
        total_gen_loss1 = 0
        total_gen_loss3 = 0
        epo_gen_loss1 = 0
        epo_gen_loss3 = 0
        current_accumulation_steps = 0
        max_patience = 10
        
               
        for epoch in tqdm(range(epoch_num)):
            for batch_gen in training_loader:
                                      
                step +=1
                
                # training the generator
                
                self.model.train()  
                batch_gen = {k: v.to(self.device) for k, v in batch_gen.items()  if k != 'intnt_lst'}
                # Forward pass
                outputs = self.model(input_ids=batch_gen['input_ids'], 
                                attention_mask=batch_gen['attention_mask'], 
                                labels=batch_gen['labels'],
                                decoder_input_ids = batch_gen['decoder_input_ids'])
                loss1 = outputs.loss
                loss1 = torch.mean(loss1) 
               
                loss3 =  torch.tensor([0])
                total_gen_loss = loss1 
                    
                total_gen_loss.backward()
                current_accumulation_steps += 1
                
                        
                total_gen_loss1 += loss1.item()
                epo_gen_loss1 += loss1.item()/ len(training_loader)
                avg_gen_loss1 = total_gen_loss1/step * grc_steps
                
                
                total_gen_loss3 += loss3.item()
                epo_gen_loss3 += loss3.item()/ len(training_loader)
                avg_gen_loss3 = total_gen_loss3/step * grc_steps
                
                if current_accumulation_steps % grc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    current_accumulation_steps = 0
            

                if step % (50 * grc_steps) == 0:
                    print(f'--------------------{step}th iter ------------- ')
                    val_loss1_avg, val_loss3_avg = self.validate_gen(loader = val_loader)
                
                    print(f'Report Gen: the loss1: ({avg_gen_loss1},{val_loss1_avg});the loss3: ({avg_gen_loss3},{val_loss3_avg})')
                    
                    save_file_path = f'../refiner/{save_folder}/checkpoint-{int(step/grc_steps)}'

                    log_losses = {'Train_loss_gloss1':avg_gen_loss1, 'Val_loss_gloss1':val_loss1_avg,
                                'Train_loss_gloss3_cl':avg_gen_loss3, 'Val_loss_gloss3_cl':val_loss3_avg}
                    

                    # Save the model only if it's wrapped with DataParallel
                    if val_loss3_avg < self.best_val_loss:
                        self.best_val_loss = val_loss3_avg
                        print(f"New best validation loss: {val_loss3_avg}. Saving model.")
                        if self.previous_ckpt_path and os.path.exists(self.previous_ckpt_path):
                            shutil.rmtree(self.previous_ckpt_path)
                            print(f"Removed previous checkpoint directory: {self.previous_ckpt_path}")

                        if isinstance(self.model, torch.nn.DataParallel):
                            self.model.module.save_pretrained(save_file_path)
                        else:
                            # Handle the case where DataParallel is not used
                            print("Model is not wrapped with DataParallel. Saving the model without DataParallel.")
                            self.model.save_pretrained(save_file_path)
                        self.tokenizer.save_pretrained(save_file_path)  
                        self.previous_ckpt_path = save_file_path                          
                        self.patience = 0
                    else:
                        self.patience += 1
                        if self.patience > max_patience:
                            print(f'Early stopping: 10 times in a row that the performance did not improve.')
                            break
                            
            if self.patience >= max_patience:
                break              
                         
            print(f'--------------------{epoch+1}th epoch finished------------- ')
            print(f'This is the {step}th step.')
            print(f'Report Gen: the {epoch+1}th epoch loss1: ({epo_gen_loss1},{val_loss1_avg}); the loss3: ({epo_gen_loss3},{val_loss3_avg})')                
            log_losses_epo = {'Train_loss_gloss1_epo':epo_gen_loss1,
                            'Train_loss_gloss3_cl_epo':epo_gen_loss3}
        
            epo_gen_loss1 = 0
            epo_gen_loss3 = 0

        # Record the end time
        end_time = time.time()

        # Calculate the total time spent
        total_time = end_time - start_time

        print(f"Total time spent: {total_time:.2f} seconds")    







import re

def extract_number_from_path(path):
    pattern = r'_(\d+)sd\.csv'
    
    # Search for the pattern in the path
    match = re.search(pattern, path)
    
    # If a match is found, return the captured group as an integer
    if match:
        return int(match.group(1))
    else:
        raise ValueError("No matching pattern found in the path")








parser = argparse.ArgumentParser(description='To train the refiner')
parser.add_argument('--train_epo_num', default =  6 ,type=int, help='train_epo_num')
parser.add_argument('--input_ut', default =  1 ,type=int, help='input num of uts')
parser.add_argument('--output_ut', default =  1 ,type=int, help='output num of uts')
parser.add_argument('--grc_steps', default =  3 ,type=int, help='gradient accumulation steps')
parser.add_argument('--model_name', default = 'google/flan-t5-large' ,type=str)
parser.add_argument('--train_batch_size', default =  24 ,type=int)
parser.add_argument('--llm_data_filepath', default = '../data/sgd/zphr_sgd_9sd.csv',type=str)
parser.add_argument('--train_refiner', default = 'yes',type=str)
parser.add_argument('--is_peft', default = 'yes',type=str)
parser.add_argument('--lora_r', default =  16 ,type=int)
parser.add_argument('--domain_map_filepath', type=str)

args = parser.parse_args()
train_epo_num = args.train_epo_num
input_ut = args.input_ut
output_ut = args.output_ut
grc_steps = args.grc_steps  
model_name = args.model_name
llm_data_filepath = args.llm_data_filepath
train_refiner = args.train_refiner
is_peft = args.is_peft
lora_r = args.lora_r
domain_map_filepath = args.domain_map_filepath


lr = 1e-4
train_batch_size = args.train_batch_size
val_batch_size = 24

device = 'cuda' if cuda.is_available() else 'cpu'
seed_value = extract_number_from_path(path = llm_data_filepath)
model_name_tail = model_name.split('/')[-1]




if 'sgd' in llm_data_filepath:
    dataset_name = 'sgd'
    real_data_filepath = '../data/sgd/train_dev_test.csv'
    
elif 'clinc150' in llm_data_filepath:
    dataset_name = 'clinc150'
    real_data_filepath = '../data/all_rd_train.csv'
    
llm_data_filename = os.path.splitext(os.path.basename(llm_data_filepath))[0]  


if is_peft == 'yes':  
    save_folder = f'{llm_data_filename}_{model_name_tail}_Peft_{dataset_name}_ut{input_ut}2ut{output_ut}'
elif is_peft == 'no':
    save_folder = f'{llm_data_filename}_{model_name_tail}_{dataset_name}_ut{input_ut}2ut{output_ut}'
print(f'device: {device}; input_ut:{input_ut}; output_ut:{output_ut}; grc_steps:{grc_steps}; model: {model_name_tail}; seed:{seed_value}')
print(f'save file name: {save_folder}')






# Reproducibility
torch.manual_seed(seed_value)
np.random.seed( seed_value) 
torch.backends.cudnn.deterministic = True
set_seed(seed_value)
random.seed(seed_value)  
        
# load generator

if is_peft == 'yes':
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r = lora_r, lora_alpha=32, lora_dropout =  0.1)
    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    model = model.to(device)
elif is_peft == 'no':
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(['<sep>'])





refiner = TrainRefiner(dataset_name = dataset_name, 
                       model = model, tokenizer = tokenizer, device = device,
                       input_ut = input_ut, output_ut = output_ut, seed = seed_value)

refiner.load_all_data(llm_data_filepath = llm_data_filepath, 
                      real_data_filepath = real_data_filepath, 
                      domain_map_filepath = domain_map_filepath)
refiner.get_train_valid_eval_domain_split()
refiner.train_model_val()

if train_refiner == 'yes':
    
    refiner.process_and_tokenize_loaded_data()
    refiner.modeling(grc_steps = grc_steps, epoch_num = train_epo_num, lr = lr)


