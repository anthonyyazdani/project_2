from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
import numpy as np

def tokenize(text_file):
    tokenized_file = [word_tokenize(i) for i in tqdm(text_file)]
    return tokenized_file

def dict_embed(tokenized_file, embeddings_dict, dim):
    
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