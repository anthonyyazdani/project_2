from trivial_data_processing_functions import *
import os
import json
data_dir = os.path.dirname(os.getcwd()) + "\\raw_data"

# Importing tweets
pos = open(data_dir + "\\train_pos_full.txt", "r", encoding="utf8").readlines()
neg = open(data_dir + "\\train_neg_full.txt", "r", encoding="utf8").readlines()
test = open(data_dir + "\\test_data.txt", "r", encoding="utf8").readlines()

# Keeping unique tweets for training data
pos = list(set(pos))
neg = list(set(neg))

# Processing tweets
pos = process_text_trivial(pos)
neg = process_text_trivial(neg)
test = process_text_trivial(test)

# Saving results
with open('trivial_positive.txt', 'w') as filehandle:
    json.dump(pos, filehandle)
    
with open('trivial_negative.txt', 'w') as filehandle:
    json.dump(neg, filehandle)
    
with open('trivial_test.txt', 'w') as filehandle:
    json.dump(test, filehandle)