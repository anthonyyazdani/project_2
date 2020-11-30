from tqdm.auto import tqdm
import re

def delete_user_rt_n_url(text):
    output = re.sub(r"<user>", "", text)
    output = re.sub(r"rt", "", output)
    output = re.sub(r"\n", "", output)
    output = re.sub(r"<url>", "", output)

    return output

def delete_non_alphabetic(text):
    text = re.sub("[^A-Za-z]", " ", text)

    return text

def standardize(text):
    text = text.lower()
    text = text.split()
    text = (" ").join(text)

    return(text)

def process_text_trivial(text_file):
    OUT = list()
    for i in tqdm(text_file):
        i = delete_user_rt_n_url(i)
        i = delete_non_alphabetic(i)
        i = standardize(i)
        if len(i)<=0:
            i = "nan"
        OUT.append(i)
    return OUT