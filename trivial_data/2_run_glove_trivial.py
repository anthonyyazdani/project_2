from glove_functions import *
import os
import time
import json

####################
embedding_size = 200
vocab_size = 20000
num_epoch = 100
batch_size = 512
learning_rate = 0.0001
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


start = time.time()

model.train(num_epoch=num_epoch, batch_size=batch_size,
            learning_rate=learning_rate, early_stopping=early_stopping)
embeddings, traductions = model.get_embeddings()
print("\n")

end = time.time()
temp = end-start
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print("Elapsed time :", '%d:%d:%d' %(hours,minutes,seconds))


# Saving results
print("----------------------------------------")
print("Saving results...")
with open(SAVING_DIR + '\\embeddings\\glove\\glove_200d.txt', 'w', encoding='utf-8') as filehandle:
    for listitem in traductions:
        filehandle.write('%s\n' % listitem)
print("Saving done !")