import json
import csv
import os
import argparse

def data_filter(ut):
    """
    Get rid of list numbers.
    """
    numbers = [f'{i}.' for i in range(1,31)]
    for num in numbers:
        if num in ut:
            return True
    return False    

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def save_to_csv(data, data_filename,dataset_name):
  
    if 'zphr' in data_filename:
        save_path = f'../data/{dataset_name}/score/supergen_{data_filename}.csv'
    elif 'llama3' in data_filename:
        save_path = f'../data/{dataset_name}/score/supergen_{data_filename}.csv'
    elif 'gpt' in data_filename:
        save_path = f'../data/{dataset_name}/score/supergen_{data_filename}.csv'       

    with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    
    # Write each inner list as a separate row
        csvwriter.writerows(data)

    print(f"The result has been written to {save_path}")



parser = argparse.ArgumentParser(description='To evaluate the refined data')
parser.add_argument('--filepath', default = 'clinc150',type=str)


args = parser.parse_args()
filepath = args.filepath
print(f'import filepath is: {filepath}')
data_filename = os.path.splitext(os.path.basename(filepath))[0]  

if 'sgd' in filepath:
    dataset_name = 'sgd'
    sample_size = 200
elif 'clinc150' in filepath:
    dataset_name = 'clinc150'
    sample_size = 100

data = load_json(filepath)


intents_list = list(data.keys())

sorted_data = dict()
for intent, example in data.items():
    sorted_example = sorted(example, key=lambda x: x[1], reverse=True)
    sorted_data[intent] = sorted_example

sorted_score_demo = [exp[1] for exp in sorted_example]
sorted_score_demo_length = len(sorted_score_demo)
print(f'sorted_score_demo: {sorted_score_demo[:5]}, sorted_score_demo_length: {sorted_score_demo_length}')

result = []
for intent, example in sorted_data.items():
    count = 0
    for exp in example:
        pred = exp[0]
        pred = pred.splitlines()
        pred = [item for item in pred if item.strip()]
        int_ut_gen = [[ut.split('. ', 1)[-1].replace('"', '').strip() , intent] for ut in pred if data_filter(ut)]
        num = int(sample_size-count)
        num_exp  = int_ut_gen[:num]
        result += num_exp
        count += len(num_exp)
        if count == 200:
            break
print(f'number of intents: {len(sorted_data.keys())}; number of examples: {len(result)}')

save_to_csv(result, data_filename,dataset_name)