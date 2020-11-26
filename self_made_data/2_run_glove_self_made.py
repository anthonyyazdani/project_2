from glove_functions import *
import os
import json

####################
embedding_size = 200
vocab_size = 100
num_epoch = 500
####################


# Get path

DATA_DIR = os.path.dirname(os.getcwd())
SAVING_DIR = os.path.dirname(os.getcwd())

# Load data

data, labels, test = load_data(DATA_DIR + '\\self_made_data\\self_made_positive.txt',
                               DATA_DIR + '\\self_made_data\\self_made_negative.txt',
                               DATA_DIR + '\\self_made_data\\self_made_test.txt',
                               subset=False)

# Get full text in one file

full = data + test


if torch.cuda.is_available():
    model = GLOVE(embedding_size, vocab_size).cuda()
else:
    model = GLOVE(embedding_size, vocab_size)

model.fit(full)
model.train(num_epoch=num_epoch)
embeddings, traductions = model.get_embeddings()

# Saving results

with open(SAVING_DIR + '\\embeddings\\glove\\glove_self_made_200d.txt', 'w') as filehandle:
    for listitem in traductions:
        filehandle.write('%s\n' % listitem)