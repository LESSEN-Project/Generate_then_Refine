from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import csv
import json
import random
from tqdm import tqdm
import argparse
import re
from openai import OpenAI
import numpy as np
from torch import cuda


class GeneratedData:
    
    def __init__(self, model_name, dataset_name, seed_value):
        allowed_datasets = ['sgd','clinc150']
        assert dataset_name in allowed_datasets, f"dataset_name must be one of {allowed_datasets}"
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.seed_value = seed_value
        self.intent_dm_dic = None
        self.dm_intent_dic = None
        self.intent_list = None
        self.prompt = None
        self.data = None
        

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
        
    def generate_examples_prompt(self, intent, exp_num = 6):
        # Convert underscore to space for better readability
        category_name = self.adjust_intents(input_string = intent)
        dm = self.intent_dm_dic[intent]
        if exp_num ==1:
            prompt = f"Give me {exp_num} user's utterance indicating the user intent of \"{category_name}\" in the \"{dm}\" context."
        else:    
            prompt = f"Give me {exp_num} user's utterances indicating the user intent of \"{category_name}\" in the \"{dm}\" context."
        self.prompt = prompt
    
    
        

    def get_intent_domain_map(self, file_path):
        """
        Load a dictionary from a json file. In our specefic case, we will load domain, intent pair
        Input:
        - file_path (str): The file path of the json file to be loaded.
        The method will update the three self attributes 
        """
        with open(file_path, 'r') as file:
            # Load the JSON data from the file
            dm_intent_dic = json.load(file)
        intent_dm_dic = dict()
        for dm in dm_intent_dic:
            intent_list = dm_intent_dic[dm]
            for intent in intent_list:
                intent_dm_dic[intent] = dm
        self.dm_intent_dic = dm_intent_dic
        self.intent_dm_dic = intent_dm_dic
        self.intent_list =  list(self.intent_dm_dic.keys())
        print(f'There are totaling {len(self.intent_list)} elements in the intent list.')
    
    @staticmethod
    def valid_average(tensor):
        valid_numbers = tensor[torch.isfinite(tensor)]
        if valid_numbers.numel() == 0:
            return float('nan')  # Return NaN if there are no valid numbers
        return torch.mean(valid_numbers.float()).item()    

    @staticmethod
    def data_filter(ut):
        """
        Get rid of list numbers.
        """
        numbers = [f'{i}.' for i in range(1,31)]
        for num in numbers:
            if num in ut:
                return True
        return False    
    def generate_utterance(self, domain_map_filepath, num_sequences = 30):
        model_name = self.model_name
        transformers.set_seed(self.seed_value)
        pipeline = transformers.pipeline("text-generation", model= model_name,torch_dtype=torch.bfloat16, device_map="auto")
        prompt_rslt = []
        self.get_intent_domain_map(file_path = domain_map_filepath)
        for intent in tqdm(self.intent_list):
            self.generate_examples_prompt(intent = intent, exp_num =6)  
            messages = [
                {"role": "system",
                    "content": "You are a chatbot that always answers with accurate responses",},
                {"role": "user", "content": self.prompt},
            ]
            prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)      
            if 'zephyr-7b-beta' in model_name:
                sequences = pipeline(prompt, max_new_tokens= 420, num_return_sequences= num_sequences, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            elif 'Meta-Llama-3' in model_name:
                terminators = [pipeline.tokenizer.eos_token_id,pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                sequences = pipeline(prompt, eos_token_id=terminators, max_new_tokens= 420, num_return_sequences= num_sequences, do_sample=True, temperature=0.6, top_k=50, top_p=0.9)
                
             
            print(sequences[0]["generated_text"])
                
            for seq in sequences:
                
                pred = seq["generated_text"][len(prompt):]  
                pred = pred.splitlines()
                # Remove empty strings from the list
                pred = [item for item in pred if item.strip()]
                
                # make it into [ ut, intnt]
                int_ut_gen = [[ut.split('. ', 1)[-1].replace('"', '').strip() , intent] for ut in pred if GeneratedData.data_filter(ut)]
                
                prompt_rslt+= int_ut_gen
         
        self.data = prompt_rslt

    def generate_utterance_Chatgpt(self, domain_map_filepath, num_sequences = 30):
        prompt_rslt = []
        self.get_intent_domain_map(file_path = domain_map_filepath)
        for intent in tqdm(self.intent_list):
            self.generate_examples_prompt(intent = intent, exp_num =6)  
            
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": "You are a chatbot that always answers with accurate responses"},
            {"role": "user", "content": self.prompt}],
            max_tokens=420,
            n=num_sequences)
            print(f'prompt is: {self.prompt}')
            for choice in response.choices:
                pred = choice.message.content
                pred = pred.splitlines()
                pred = [item for item in pred if item.strip()]
                int_ut_gen = [[ut.split('. ', 1)[-1].replace('"', '').strip() , intent] for ut in pred if GeneratedData.data_filter(ut)]
                prompt_rslt+= int_ut_gen
            
        self.data = prompt_rslt
        
        
    def generate_utterance_scores(self, domain_map_filepath):
        verbose = 1
        transformers.set_seed(self.seed_value)
        model_name = self.model_name
        device = 'cuda' if cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)
        model = model.to(device)
        result = dict()
        self.get_intent_domain_map(file_path = domain_map_filepath)
        
        if self.dataset_name == 'clinc150':
            num_rep = 3
        elif self.dataset_name == 'sgd':
            num_rep = 6
        for intent in tqdm(self.intent_list):
            self.generate_examples_prompt(intent = intent, exp_num = 6)  
            messages = [
                {"role": "system",
                    "content": "You are a chatbot that always answers with accurate responses",},
                {"role": "user", "content": self.prompt},
            ]
            
            prompt = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt")
            prompt = prompt.to(device)
            for i in range(num_rep):
                if 'zephyr-7b-beta' in model_name:
                    outputs = model.generate(prompt,max_new_tokens= 420, num_return_sequences= 60, return_dict_in_generate=True, do_sample=True, output_scores=True, temperature=0.7, top_k=50, top_p=0.95)

                elif 'Meta-Llama-3' in model_name:
                    terminators = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                    outputs = model.generate(prompt,max_new_tokens= 420, num_return_sequences= 60, return_dict_in_generate=True, do_sample=True, output_scores=True, temperature=0.6, top_k=50, top_p=0.9, eos_token_id=terminators)                    
                    
                model = model.to('cpu')
                outputs_scores_cpu = tuple(t.cpu() for t in outputs.scores)
                outputs_sequences_cpu = outputs.sequences.to('cpu')
                transition_scores = model.compute_transition_scores(outputs_sequences_cpu, outputs_scores_cpu, normalize_logits=True)

                generated_tokens = outputs.sequences  
                if verbose == 1: #demo
                    print(f'Prompt:\n{self.prompt}')
                    print(f"Get one demo:")
                    print(f'generated_tokens shape {generated_tokens.shape}')
                    print(f"Prompt shape: {prompt.shape}")
                    print(f'transition_scores shape {transition_scores.shape}')
                    print('Prompt:\n',tokenizer.decode(generated_tokens[0][:len(prompt[0])]))
                    dmo_out = generated_tokens[0][len(prompt[0]):]
                    print('Output:\n',tokenizer.decode(dmo_out))
                     
                num_of_sequence = len(transition_scores)
                for i in range(num_of_sequence):
                    one_pred_score = transition_scores[i]

                    pred = tokenizer.decode(generated_tokens[i][len(prompt[0]):], skip_special_tokens=True) 
                    score_avg = GeneratedData.valid_average(one_pred_score)
                    a_row = [pred,score_avg]
                    if intent not in result:
                        result[intent] = [a_row]
                    else:
                        result[intent].append(a_row)
                    if verbose == 1: 
                        for tok, score in zip(generated_tokens[0][len(prompt[0]):], transition_scores[0]): # demo
                            print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
                            verbose += 1
                del outputs
                model = model.to(device)
                 
               
        self.data = result
        
                             
    def save_to_csv(self):
        """
        Save the prompt result dictionary to a JSON file.
        Input:
        - data: The dictionary to be saved.
        - file_path: The file path where the JSON file will be saved.
        """


        
        if 'zephyr-7b-beta' in self.model_name:
            save_path = f'../data/{self.dataset_name}/zphr_{self.dataset_name}_{seed_value}sd.csv'
        elif 'Meta-Llama-3-70B-Instruct' in self.model_name:
            save_path = f'../data/{self.dataset_name}/llama3Large_{self.dataset_name}_{seed_value}sd.csv'
        elif 'Meta-Llama-3-8B-Instruct' in self.model_name:
            save_path = f'../data/{self.dataset_name}/llama3_{self.dataset_name}_{seed_value}sd.csv'
        elif 'gpt' in self.model_name:
            save_path = f'../data/{self.dataset_name}/chatgpt_{self.dataset_name}_{seed_value}sd.csv'       

        with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        
        # Write each inner list as a separate row
            csvwriter.writerows(self.data)

        print(f"The result has been written to {save_path}")
        

    def save_dict_to_json(self):
        if 'zephyr-7b-beta' in self.model_name:
            save_path = f'../data/{self.dataset_name}/score/zphr_{self.dataset_name}_{self.seed_value}sd.json'
        elif 'Meta-Llama-3-70B-Instruct' in self.model_name:
            save_path = f'../data/{self.dataset_name}/score/llama3Large_{self.dataset_name}_{self.seed_value}sd.json'
        elif 'Meta-Llama-3-8B-Instruct' in self.model_name:
            save_path = f'../data/{self.dataset_name}/score/llama3_{self.dataset_name}_{self.seed_value}sd.json'
        with open(save_path, 'w') as json_file:
            json.dump(self.data, json_file)
        print(f"The result has been written to {save_path}")



parser = argparse.ArgumentParser(description='Prompting LLM to get generated utterance')
parser.add_argument('--seed_value', type=int,default = 1)
parser.add_argument('--dataset_name', type=str, default = 'clinc150')
parser.add_argument('--model_name', type=str, default = 'Llama-3-8B', help ='Model used for gerenated utterances')
parser.add_argument('--get_score', type=str, default = 'no', help ='to get the output confidece. It is used to reproduce SuperGen data selectoin')
parser.add_argument('--domain_map_filepath', type=str)


args = parser.parse_args()
seed_value = args.seed_value
dataset_name = args.dataset_name
model_name = args.model_name
get_score = args.get_score
domain_map_filepath = args.domain_map_filepath
client = OpenAI(
    api_key= None # This api_key is for Chatgpt
) 

if model_name in 'zephyr-7b-beta':
    model_name = 'HuggingFaceH4/zephyr-7b-beta'
elif model_name in 'Meta-Llama-3-70B-Instruct':
    model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
elif model_name in 'Meta-Llama-3-8B-Instruct':
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
elif model_name in 'gpt-3.5-turbo':
    model_name = 'gpt-3.5-turbo'

if get_score == 'yes':
    num_gen = 7
else:
    num_gen = 1



if dataset_name == 'clinc150':
    num_sequences = 30 * num_gen
elif dataset_name == 'sgd':
    num_sequences = 50 *num_gen

print(f"seed_value: {seed_value}; dataset_name: {dataset_name}; model: {model_name}; num_sequences: {num_sequences}")


llm_gen = GeneratedData(model_name = model_name, dataset_name = dataset_name, seed_value = seed_value)
if model_name == 'gpt-3.5-turbo':
    llm_gen.generate_utterance_Chatgpt(domain_map_filepath = domain_map_filepath, num_sequences = num_sequences)
    llm_gen.save_to_csv()
elif get_score == 'yes':
    llm_gen.generate_utterance_scores(domain_map_filepath = domain_map_filepath)
    llm_gen.save_dict_to_json()
else:
    llm_gen.generate_utterance(domain_map_filepath = domain_map_filepath, num_sequences = num_sequences)
    llm_gen.save_to_csv()

 