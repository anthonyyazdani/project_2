from functions.nn_functions import *
from functions.classical_ML_functions import *

#The procedure is the following:
#1. We use the "trivial details" where we deleted labels and tags such as user> and url>, non-alphabetic words and white spaces. Also, we put the words to lower case. This procedure is done in the file "1_run_trivial_dtp.py" for the train set and the test set.
#2. We use pre-trained GloVe embeddings from Stanford of size 200 for the 20 000 most frequent words to make our predictions.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1997)
torch.backends.cudnn.deterministic = True
vocab_size = 20000
batch_size = 64
embedding_dimension = 200

#Training

text = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
label = data.LabelField(dtype=torch.float)
fields = [('text', text), ('label', label)]

vectors = Vectors(name='embeddings/stanford_glove/glove.twitter.27B.200d.txt')

training, validation = data.TabularDataset.splits(path='trivial_data',
                                                  train='train_trivial.csv',
                                                  validation="val_trivial.csv",
                                                  format='csv',
                                                  fields=fields,
                                                  skip_header=True)

text.build_vocab(training,
                 max_size=vocab_size,
                 vectors=vectors,
                 unk_init=torch.Tensor.normal_)

label.build_vocab(training)

batch_bucket, batch_valid_bucket = data.BucketIterator.splits((training, validation),
                                                              batch_size=batch_size,
                                                              device=device,
                                                              sort=False)

model_trivial = BI_LSTM(vocabulary_size=len(text.vocab),
                        embedding_dimension=embedding_dimension)
model_trivial.embedding.weight.data.copy_(text.vocab.vectors)
optimizer = optim.Adam(model_trivial.parameters())
criterion = nn.BCEWithLogitsLoss()

train(model_trivial, 3, batch_bucket, batch_valid_bucket, optimizer, criterion)


#Predicting

_, _, test_trivial = load_data('trivial_data/trivial_positive.txt',
                               'trivial_data/trivial_negative.txt',
                               'trivial_data/trivial_test.txt')
_, lstm_submission = predict(text, model_trivial, test_trivial)
create_csv_submission(
    lstm_submission, 'submission_lstm_model_sg_t.csv')
