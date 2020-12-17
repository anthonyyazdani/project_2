#We would like to thank all contributors from https://github.com/bentrevett/pytorch-sentiment-analysis which guided us in the implementation of those neural network.

import torch
from torchtext import data
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import Vectors
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np


class BI_LSTM(nn.Module):
    "Bidirectional LSTM model."

    def __init__(self, vocabulary_size, embedding_dimension, hidden_dimension=256, num_layers=2, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dimension, padding_idx=1)

        self.rnn = nn.LSTM(embedding_dimension,
                           hidden_dimension,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=True)

        self.fc = nn.Linear(hidden_dimension * 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed_embedded)
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc(hidden)

        return output


def accuracy(predicted, reference):
    """
    Compute the ratio of correctly predicted labels.
    """
    labels = torch.round(torch.sigmoid(predicted))
    correct_predictions = (labels == reference).float()

    return correct_predictions.sum().float() / correct_predictions.nelement()


def single_epoch_train(model, bucket_iterator, optimizer, criterion):
    """
    Compute forward pass and do a step in the direction of the negative gradient, ones.
    """
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    # Progress bar
    numerator = 0

    for batch in bucket_iterator:
        # Progress bar
        numerator += 1
        perc = round((numerator/len(bucket_iterator))*100)
        optimizer.zero_grad()
        text, text_lengths = batch.text
        text_lengths = torch.as_tensor(
            text_lengths, dtype=torch.int64, device='cpu')
        predictions = model(text.T, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        print(f"Epoch progress : {perc}%                 \r", end="")
    print(f"                                                                    \r", end="")

    return epoch_loss/len(bucket_iterator), epoch_acc/len(bucket_iterator)


def validation_test(model, bucket_iterator, criterion):
    """
    Evaluate the model on the validation set.
    """
    epoch_loss = 0
    epoch_accuracy = 0
    model.eval()

    with torch.no_grad():

        for batch in bucket_iterator:

            text, text_lengths = batch.text
            text_lengths = torch.as_tensor(
                text_lengths, dtype=torch.int64, device='cpu')
            predictions = model(text.T, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_accuracy += acc.item()

    return epoch_loss/len(bucket_iterator), epoch_accuracy/len(bucket_iterator)


def train(model, num_epoch, batch_iterator, batch_valid_iterator, optimizer, criterion):
    """
    Apply single_epoch_train(), <num_epoch> times.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(num_epoch):

        train_loss, train_accuracy = single_epoch_train(
            model, batch_iterator, optimizer, criterion)
        valid_loss, valid_accuracy = validation_test(
            model, batch_valid_iterator, criterion)
        print('')
        print(f'Epoch {epoch+1}')
        print(
            f'\tTrain :      Loss = {round(train_loss, 3)}, Accuracy = {round(train_accuracy*100, 2)}%')
        print(
            f'\tValidation : Loss = {round(valid_loss, 3)}, Accuracy = {round(valid_accuracy*100, 2)}%')
        print(f'-----------------------------------------------------')


def single_epoch_train_cnn(model, bucket_iterator, optimizer, criterion):
    """
    Compute forward pass and do a step in the direction of the negative gradient, ones.
    """
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    # Progress bar
    numerator = 0

    for batch in bucket_iterator:
        # Progress bar
        numerator += 1
        perc = round((numerator/len(bucket_iterator))*100)
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        print(f"Epoch progress : {perc}%                 \r", end="")
    print(f"                                                                    \r", end="")

    return epoch_loss/len(bucket_iterator), epoch_acc/len(bucket_iterator)


def validation_test_cnn(model, bucket_iterator, criterion):
    """
    Evaluate the model on the validation set.
    """
    epoch_loss = 0
    epoch_accuracy = 0
    model.eval()

    with torch.no_grad():

        for batch in bucket_iterator:

            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_accuracy += acc.item()

    return epoch_loss/len(bucket_iterator), epoch_accuracy/len(bucket_iterator)


def train_cnn(model, num_epoch, batch_iterator, batch_valid_iterator, optimizer, criterion):
    """
    Apply single_epoch_train_cnn(), <num_epoch> times.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(num_epoch):

        train_loss, train_accuracy = single_epoch_train_cnn(
            model, batch_iterator, optimizer, criterion)
        valid_loss, valid_accuracy = validation_test_cnn(
            model, batch_valid_iterator, criterion)
        print('')
        print(f'Epoch {epoch+1}')
        print(
            f'\tTrain :      Loss = {round(train_loss, 3)}, Accuracy = {round(train_accuracy*100, 2)}%')
        print(
            f'\tValidation : Loss = {round(valid_loss, 3)}, Accuracy = {round(valid_accuracy*100, 2)}%')
        print(f'-----------------------------------------------------')


class D1_CNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension, num_filters, filters, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dimension, padding_idx=1)
        self.conv = nn.ModuleList([nn.Conv1d(
            in_channels=embedding_dimension, out_channels=num_filters, kernel_size=fs) for fs in filters])
        self.linear = nn.Linear(len(filters) * num_filters, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        emb = self.embedding(text)
        emb = emb.permute(0, 2, 1)
        convolutions = [F.relu(convolution(emb)) for convolution in self.conv]
        max_pool = [F.max_pool1d(convolution, convolution.shape[2]).squeeze(2)
                    for convolution in convolutions]
        pred = self.dropout(torch.cat(max_pool, dim=1))
        output = self.linear(pred)
        return output


def predict_one_sentence(text_build_vocab, model, sentence):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq = sentence.lower().split()
    index = [[text_build_vocab.vocab.stoi[i]] for i in seq]
    length = torch.tensor([len(index)]).long()
    length = torch.as_tensor(length, dtype=torch.int64, device='cpu')
    index = torch.tensor(index).long().to(device)
    pred = model(index, length)
    return torch.sigmoid(pred).item()


def predict(text_build_vocab, model, text):

    probability = np.array([predict_one_sentence(text_build_vocab, model, i) for i in tqdm(text)])
    preds = np.round_(probability)
    preds[np.where(preds == 0)] = -1

    return probability, preds


def predict_one_sentence_cnn(text_build_vocab, model, sentence, max_filter_size):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq = sentence.lower().split()
    model.eval()
    if len(seq) < max_filter_size:
        for i in range(max_filter_size-len(seq)):
            seq += ['<pad>']
    index = [text_build_vocab.vocab.stoi[i] for i in seq]
    index = torch.tensor(index).long().to(device).unsqueeze(0)
    pred = model(index)

    return torch.sigmoid(pred).item()


def predict_cnn(text_build_vocab, model, text, max_filter_size):

    probability = np.array([predict_one_sentence_cnn(text_build_vocab, model, i, max_filter_size) for i in tqdm(text)])
    preds = np.round_(probability)
    preds[np.where(preds == 0)] = -1

    return probability, preds
