from glove_functions import *
import os
import json

####################
embedding_size = 200
vocab_size = 3000
num_epoch = 100
batch_size = 512
learning_rate = 0.001
early_stopping = 3
####################


# Get path

DATA_DIR = os.path.dirname(os.getcwd())
SAVING_DIR = os.path.dirname(os.getcwd())

# Load data
print("----------------------------------------")
print("Loading data...")
data, labels, test = load_data(DATA_DIR + '\\trivial_data\\trivial_positive.txt',
                               DATA_DIR + '\\trivial_data\\trivial_negative.txt',
                               DATA_DIR + '\\trivial_data\\trivial_test.txt',
                               subset=False)

# Get full text in one file

full = data + test
print("Loading done !")
print("\n")

if torch.cuda.is_available():
    model = GLOVE(embedding_size, vocab_size).cuda()
else:
    model = GLOVE(embedding_size, vocab_size)

print("----------------------------------------")
print("Processing data...")
model.fit(full)
print("Processing done !")
print("\n")

print("----------------------------------------")
print("Beginning of training...")
model.train(num_epoch=num_epoch, batch_size=batch_size,
            learning_rate=learning_rate, early_stopping=early_stopping)
embeddings, traductions = model.get_embeddings()
print("\n")

# Saving results
print("----------------------------------------")
print("Saving results...")
with open(SAVING_DIR + '\\embeddings\\glove\\glove_trivial_200d.txt', 'w') as filehandle:
    for listitem in traductions:
        filehandle.write('%s\n' % listitem)
print("Saving done !")