# Generate_then_Refine

## Clinc150 and SGD datasets:
This project processes datasets from the following sources:


**Clinc150**

You can download the dataset from the following link:

https://github.com/clinc/oos-eval

**SGD**

You can download the dataset from the following link:
 
https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

We use the source code belowe to process SGD dataset

https://github.com/RIDE-SIGIR/GZS

## The script folder 
It includes whole pipeline, start with generating the syhthetic utterances (01prompting.py), training refiner(02refiner_train.py), generate refine utterances (03refiner_infer.py), and evaluation (04refiner_eval_dist.py and 04refiner_eval_acc.py)


### note:
01-1get_SuperGen.py: When you choose to generate based on output confidence from 01prompting.py (i.e. --get_score yes). Then the output will be a json file, 01-1get_SuperGen.py will take the json file as input and then dervie the most confidece output as a csv file. 

## How to use
For each script, there is a **domain_map_filepath** argument. This should be a dictionary where the keys are domains and the values are lists of intents, saved as a JSON file. For **Clinc150**, the json file is provided in the source, but for **SGD**, you have to create it yourself.

For 01prompting.py, in line 280, there is an openAI key need to be put if you want to reproduce ChatGPT result. Otherwise you can comment it out and its related function/method.

For 02refiner_train.py, the **real_data_filepath** variable  should point to the default training data CSV file path **Clinc150**, and the complete train-dev-test split CSV file for **SGD**. As mentioned in our paper, we merge all splits and perform our own split to enhance generalizability. Each row in the CSV file should contain utterance, intent, and domain.

For 04refiner_eval_acc.py, the **real_test_data_filepath** variable should point to the default test data CSV file path for **Clinc150**, and the complete train-dev-test split CSV file for **SGD**. Each row in the CSV file should also contain utterance, intent, and domain.


Once you have set up the necessary files and arguments, you're good to go!
