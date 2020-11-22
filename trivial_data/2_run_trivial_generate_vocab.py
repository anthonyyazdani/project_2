import os
import numpy as np
from tqdm.auto import tqdm
import json

# Importing files
with open('trivial_positive.txt', 'r', encoding='utf-8') as filehandle:
    pos = json.load(filehandle)

with open('trivial_negative.txt', 'r', encoding='utf-8') as filehandle:
    neg = json.load(filehandle)
    
with open('trivial_test.txt', 'r', encoding='utf-8') as filehandle:
    test = json.load(filehandle)

# Creating vocab
data = pos + neg + test
vocab = [word for sentence in tqdm(data) for word in sentence.split()]
vocab = list(set(vocab))

# Saving results
with open('trivial_vocab.txt', 'w') as filehandle:
    json.dump(vocab, filehandle)