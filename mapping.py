import pandas as pd
import zipfile
import re
import os
import json
import torch
import pickle
import random
import requests
import numpy as np
import datasets

from datasets import load_dataset, concatenate_datasets, load_from_disk, load_metric, Dataset, ClassLabel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split

# read zipfile
zf = zipfile.ZipFile('/content/starting_ki.zip') 
# check contained files
zf.namelist()

df = pd.read_csv(zf.open('train_all_tasks.csv'))

df['label_sexist'] = df['label_sexist'].replace('sexist', 1)
df['label_sexist'] = df['label_sexist'].replace('not sexist', 0)

# keep only the things we need
df = df[['rewire_id', 'text', 'label_sexist']]
# split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=25)

train = Dataset.from_dict(train_df)
test = Dataset.from_dict(test_df)
sexist = datasets.DatasetDict({"train":train,"test":test})


sexist_train_samples = sexist['train'].filter(lambda e: e['label_sexist'] == 1).shuffle(seed=42).select(range(10))
not_train_samples = sexist['train'].filter(lambda e: e['label_sexist'] == 0).shuffle(seed=42).select(range(10))

# map emotions to integers for labeling
def map_label(example):
    if example['label_sexist'] == 0:
        example['label_sexist'] = 0
    elif example['label_sexist'] == 1:
       example['label_sexist'] = 1
    
   

# create a train set that consists of 10 samples per class and filter the test 
# set to contain only the valid labels
sexist_train_ds = concatenate_datasets([sexist_train_samples, not_train_samples]).map(lambda e: map_label(e)).shuffle(seed=42)
sexist_test_ds = sexist["test"].filter(lambda e: e['label_sexist'] in [0,1]).map(lambda e: map_label(e))

# define the maping between emotions and labels
mapping = ClassLabel(names=['sexist', 'not sexist'])

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')

model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')

# define the number of synthetic samples to generate
n = 10
new_texts = []
new_labels = []

iter = 0
while iter < n:
    # select two random samples from training set
    text1, label1, text2, label2 = get_two_random_samples()
    # create the prompt
    prompt = get_prompt(text1, label1, text2, label2)

    # generate text using GPT-J model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    # the generated output will be in the form "<text> (Sentiment: <label>)"
    data = gen_text.split('\n')[3].strip('Post: ').split('(Label:')
    if len(data) < 2:
        # the format of the response is invalid
        continue

    text = data[0]
    label = data[1].split(')')[0].strip()
    if label not in ['sexist', 'not sexist']:
        # the format of the response is invalid
        continue

    new_texts.append(text)
    new_labels.append(mapping.str2int(label))
    iter += 1


# define the synthetic dataset and save it to disk 
synthetic_ds = Dataset.from_dict({'text': new_texts, 'label': new_labels})
print(synthetic_ds)
# synthetic_ds.save_to_disk('./data/gpt-neo/' + str(n))
