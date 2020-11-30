from nltk.corpus import opinion_lexicon
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import enchant
from autocorrect import Speller
from tqdm.auto import tqdm
import re
import numpy as np
import pandas as pd


def delete_user_rt_n_url(text):
    output = re.sub(r"<user>", "", text)
    output = re.sub(r"rt", "", output)
    output = re.sub(r"\n", "", output)
    output = re.sub(r"<url>", "", output)

    return output


def replace_smiley(text, happy_emoticons, bad_emoticons, ambiguous_emoticons):

    for i in happy_emoticons:
        text = text.replace(i, " happy ")

    for i in bad_emoticons:
        text = text.replace(i, " sad ")

    for i in ambiguous_emoticons:
        text = text.replace(i, " ")

    return text


def replace_abv(text, abv, tr_abv):

    sp_text = np.array(text.split())

    LIST = np.intersect1d(sp_text, abv)

    if len(LIST) > 0:
        for i in LIST:
            sp_text[np.where(sp_text == i)[0]] = tr_abv[np.where(abv == i)][0]

    return (" ").join(sp_text)


def decontract(text):
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    return text


def replace_repetition(text, spell=True):
    text = re.sub(r"((\w)\2{2,})", r'\2', text)
    if spell:
        spell = Speller(lang='en')
        text = spell(text)

    return text


def delete_non_alphabetic(text):
    text = re.sub("[^A-Za-z]", " ", text)

    return text


def standardize(text):
    text = text.lower()
    text = text.split()
    text = (" ").join(text)

    return(text)


def lemmatize(text):
    text = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos="v") for word in text]
    return (" ").join(words)


def delete_stop_words(text):
    text = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in text if not w in stop_words]
    
    return (" ").join(words)


def sentiment_words(text, pos_list, neg_list):
    text = np.array(text.split())
    OUT = np.zeros(len(text))
    POS_candidates = np.intersect1d(text, pos_list)
    NEG_candidates = np.intersect1d(text, neg_list)

    for i in POS_candidates:
        index = np.where(text == i)
        OUT[index] = 1

    for i in NEG_candidates:
        index = np.where(text == i)
        OUT[index] = -1

    return np.array(OUT), np.array([sum(OUT == -1), sum(OUT == 1)])


def analyze_sentiment(text_file):

    pos_list = list(set(opinion_lexicon.positive()))
    neg_list = list(set(opinion_lexicon.negative()))

    SEQ = list()
    SUM = list()
    for sentence in tqdm(text_file):
        a = sentiment_words(sentence, pos_list, neg_list)
        SEQ.append(a[0])
        SUM.append(a[1])

    return SEQ, SUM


def process_text(text_file, spell=False):

    happy_emoticons = pd.read_csv("happy_emoticons.csv")["0"].values
    bad_emoticons = pd.read_csv("bad_emoticons.csv")["0"].values
    ambiguous_emoticons = pd.read_csv("ambiguous_emoticons.csv")["0"].values

    data = pd.read_csv("abreviation.csv")
    abv = np.array(data["abv"].values)
    tr_abv = np.array(data["translate"].values)

    for R, sentence in enumerate(tqdm(text_file)):
        temp = replace_smiley(sentence, happy_emoticons,
                              bad_emoticons, ambiguous_emoticons)
        temp = delete_user_rt_n_url(temp)
        temp = decontract(temp)
        temp = replace_repetition(temp, spell=spell)
        temp = delete_non_alphabetic(temp)
        temp = replace_abv(temp, abv, tr_abv)
        temp = standardize(temp)
        temp = delete_stop_words(temp)
        temp = lemmatize(temp)
        if len(temp)<=0:
            temp = "nan"
        text_file[R] = temp

    return text_file