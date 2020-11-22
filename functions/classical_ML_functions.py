from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import csv

def tokenize(text_file):
    
    """Transform a list of sentences to a list of tokens.""" 
    
    tokenized_file = [word_tokenize(i) for i in tqdm(text_file)]
    
    return tokenized_file


def create_dict_from_glove(path_to_glove_txt):
    
    """Creates a dictionary between words and their corresponding vector from a glove.txt file."""
    
    embeddings_index = {}
    with open(path_to_glove_txt, 'r', encoding='utf-8') as txtfile:
        lines = txtfile.readlines()
        for line in tqdm(lines):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        
    return embeddings_index


def dict_embed(tokenized_file, embeddings_dict, dim):
    
    """Creates a feature matrix with the average vectors of words in the sentences.""" 
    
    OUTPUT = np.zeros((len(tokenized_file), dim))
    for counter, sentence in enumerate(tqdm(tokenized_file)):
        temp = list()
        for word in sentence:
            candidate = embeddings_dict.get(word)
            if isinstance(candidate, np.ndarray):
                temp.append(candidate)
            else:
                pass
        if len(temp)==0:
            temp = np.zeros(dim)
        
        OUTPUT[counter] = np.mean(temp, axis = 0)
        
    return np.array(OUTPUT)



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
    
    
def stratified_kfold_LR(x, y, n_splits=5, shuffle=True, random_state=42, verbose=0, max_iter=1000):
    
    """This function performs classic stratified Kfold for logistic regression."""
    
    res = list()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for train_index, test_index in tqdm(skf.split(x, y)):

        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = LogisticRegression(verbose=verbose, max_iter=max_iter).fit(X_train, y_train)
        res.append(clf.score(X_test, y_test))

    return res


def stratified_kfold_MLP(x, y, n_splits=5, shuffle=True, random_state=42, hidden_layer_sizes=(100, 100), verbose=False, max_iter=1000, early_stopping=True):
    
    """This function performs classic stratified Kfold for multilayer perceptron."""
    
    res = list()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for train_index, test_index in tqdm(skf.split(x, y)):

        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                            random_state=1, max_iter=max_iter, verbose=verbose, early_stopping=early_stopping).fit(X_train, y_train)
        res.append(clf.score(X_test, y_test))

    return res

def stratified_kfold_NB(x, y, n_splits=5, shuffle=True, random_state=42):
    
    """This function performs classic stratified Kfold for naive bayes."""
    
    res = list()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for train_index, test_index in tqdm(skf.split(x, y)):

        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = GaussianNB().fit(X_train, y_train)
        res.append(clf.score(X_test, y_test))

    return res

def create_csv_submission(y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    ids = np.linspace(1,len(y_pred),len(y_pred)).astype(int)
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})