import json
import numpy as np
import pandas as pd

with open("trivial_positive.txt", 'r', encoding='utf-8') as filehandle:
    pos = json.load(filehandle)
    sub_pos = pos[:90233]
    
with open("trivial_negative.txt", 'r', encoding='utf-8') as filehandle:
    neg = json.load(filehandle)
    sub_neg = neg[:91088]

with open("trivial_test.txt", 'r', encoding='utf-8') as filehandle:
    test = json.load(filehandle)

data = pos + neg
sub_data = sub_pos + sub_neg

labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))), axis=0)
sub_labels = np.concatenate((np.ones(len(sub_pos)), np.zeros(len(sub_neg))), axis=0)
    
data = pd.concat([pd.DataFrame(data), pd.DataFrame(labels)], axis=1)
data = data.sample(frac=1, random_state=42)

sub_data = pd.concat([pd.DataFrame(sub_data), pd.DataFrame(sub_labels)], axis=1)
sub_data = sub_data.sample(frac=1, random_state=42)

data[:int(0.9*len(data))].to_csv("train_trivial.csv", index=False)
data[int(0.9*len(data)):].to_csv("val_trivial.csv", index=False)
pd.DataFrame(test).to_csv("test_trivial.csv", index=False)

sub_data[:int(0.9*len(sub_data))].to_csv("sub_train_trivial.csv", index=False)
sub_data[int(0.9*len(sub_data)):].to_csv("sub_val_trivial.csv", index=False)