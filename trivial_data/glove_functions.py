# We would like to thank noaRricky (https://github.com/noaRricky) for inspiring us
# in the overall structure that the implementation of a glove on pytorch takes.

import numpy as np
from itertools import combinations_with_replacement
from tqdm.auto import tqdm
from tensorflow.python.keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader, Dataset
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import json
import re


def load_data(path_to_positive_txt, path_to_negative_txt, path_to_test_txt, subset=True):
    
    """Loads the data using json. This function is inherent to the way we manage our data."""
    
    with open(path_to_positive_txt, 'r', encoding='utf-8') as filehandle:
        pos = json.load(filehandle)
        if subset:
            pos = pos[:90233]

    with open(path_to_negative_txt, 'r', encoding='utf-8') as filehandle:
        neg = json.load(filehandle)
        if subset:
            neg = neg[:91088]

    with open(path_to_test_txt, 'r', encoding='utf-8') as filehandle:
        test = json.load(filehandle)
        
    data = pos + neg
    labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))), axis=0)
    
    return data, labels, test


def cooc_mat(text, number_of_words):
    """
    Creation of the co-occurrence matrix and a dictionary that associates words with their respective indexes.
    This operation is performed on the <number_of_words> most frequent words.
    """

    # Tokenize and get most frequent words

    number_of_words = number_of_words+1
    tokenized = Tokenizer(num_words=number_of_words)
    tokenized.fit_on_texts(text)

    # Create dictionary

    dico = {v: k for k, v in dict(tokenized.index_word.items()).items()}

    # Process text

    sentences = tokenized.texts_to_sequences(text)

    # Construct co-occurrence matrix

    res = np.zeros((number_of_words, number_of_words))

    for sentence in tqdm(sentences):
        comb = combinations_with_replacement(sentence, 2)
        for i in comb:
            res[list(i)[0], list(i)[1]] += 1

    res = res + res.T - np.diag(np.diag(res))
    res = res[1:, 1:]
    res = np.array([np.kron(np.arange(len(res)), np.ones(len(res))),
                    np.outer(np.arange(len(res)),
                             np.ones(len(res))).T.flatten(),
                    res.flatten()]).T

    return res, dico


class cooc_data(Dataset):
    """
    Function to pass the coocurence matrix on the data loader.
    """

    def __init__(self, coocurrence_matrix):
        self._coocurrence_matrix = coocurrence_matrix

    def __getitem__(self, index):

        return self._coocurrence_matrix[index]

    def __len__(self):

        return len(self._coocurrence_matrix)


class GLOVE(torch.nn.Module):
    """
    Glove model.
    """

    def __init__(self, embedding_size, vocab_size, x_max=100, alpha=3/4):
        super(GLOVE, self).__init__()

        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.x_max = x_max
        self.alpha = alpha
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)
        self.word_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        self.context_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)
        self.context_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        self.dataset = None

        for params in self.parameters():
            init.uniform_(params, a=-1, b=1)

    def fit(self, text):
        """
        Fit the model on the text of interest.
        """

        coocurrence_matrix, dico = cooc_mat(text, self.vocab_size)
        self.dataset = cooc_data(coocurrence_matrix)
        self.dico = dico

    def train(self, num_epoch, batch_size=512, learning_rate=0.001):
        """
        Train the model.
        """

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        glove_dataloader = DataLoader(self.dataset, batch_size, shuffle=True)
        display_loss = 0

        for epoch in range(num_epoch):
            for idx, batch in enumerate(glove_dataloader):
                optimizer.zero_grad()

                temp = batch
                i_s, j_s, counts = temp[:, 0], temp[:, 1], temp[:, 2]
                i_s = i_s.to(device)
                j_s = j_s.to(device)
                counts = counts.to(device)
                loss = self.cost_function(i_s, j_s, counts)

                display_loss += loss.item()
                loss.backward()
                optimizer.step()
                torch.autograd.set_detect_anomaly(True)

            print("epoch: {}, loss: {}".format(epoch, display_loss))
            display_loss = 0

        print("Done !")

    def cost_function(self, word_input, context_input, coocurrence_count):
        """
        Definition of the cost function.

        For a better understanding see https://nlp.stanford.edu/pubs/glove.pdf
        """

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Parameters

        x_max = self.x_max
        alpha = self.alpha

        # Pass to embedding layers
        word_input = word_input.long()
        context_input = context_input.long()
        word_emb = self.word_embeddings(word_input)
        word_bias = self.word_biases(word_input)
        context_emb = self.context_embeddings(context_input)
        context_bias = self.context_biases(context_input)

        # Compute the loss
        equation_1 = torch.pow(coocurrence_count / x_max, alpha)
        equation_1[equation_1 > 1] = 1
        embedding_multiplication = torch.sum(word_emb * context_emb, dim=1)
        log_cooc = torch.log(coocurrence_count + 1)  # + 1 to avoid log(0)
        equation_2 = (embedding_multiplication + word_bias +
                      context_bias - log_cooc) ** 2

        # loss
        m_loss = torch.mean(equation_1 * equation_2)

        return m_loss

    def get_embeddings(self):
        """
        Extract the embeddings as in stanford's publication (sum of the 2 vectors).

        For a better understanding see https://nlp.stanford.edu/pubs/glove.pdf
        """

        w1 = np.array([list(self.word_embeddings.parameters())[
                      0][i].cpu().detach().numpy().astype(np.float32) for i in range(self.vocab_size)])
        w2 = np.array([list(self.context_embeddings.parameters())[
                      0][i].cpu().detach().numpy().astype(np.float32) for i in range(self.vocab_size)])
        embeddings = w1 + w2

        temp = [i for i in self.dico.keys()][:self.vocab_size]
        traductions = list()
        for counter, i in enumerate(temp):
            candidate = str([i, list(embeddings[counter, :])])
            candidate = re.sub("\[","",candidate)
            candidate = re.sub("]","",candidate)
            candidate = re.sub(",","",candidate)
            candidate = re.sub("'","",candidate)
            traductions.append(candidate)

        return embeddings, traductions