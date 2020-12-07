# project_2

overleaf : https://www.overleaf.com/project/5f8b59404f02e600016c7df1

TODO : **Folder structure in tree form**

### Requirements
- Minimum hardware requirements:
	- 16GB ram (32GB ram if you want to have a large number of words in your vocabulary for the glove)
	- 6GB CUDA graphic card
- Software:
	- Windows 10
- Python:
	- 3.8.3
- Libraries:
	- gensim
	- tqdm
	- numpy
	- json
	- sklearn
	- csv
	- nltk
	- itertools
	- tensorflow
	- pytorch (GPU)
	- collections
	- re
	- time
	
### Download requirements
- Stanford glove:
	- [glove.twitter.27B.200d](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
	- Add "glove.twitter.27B.200d.txt" to ~\project_2\embeddings\stanford_glove
- Dataset:
	- [data](https://www.aicrowd.com/challenges/epfl-ml-text-classification#dataset)
	- Add "train_*_full.txt" files to ~\project_2\raw_data
- nltm:
	```import nltk```
	```nltk.download('stopwords')```
	```nltk.download('wordnet')```
	```nltk.download('punkt')```
	```nltk.download("opinion_lexicon")```

### Summary

- 2 different data processing:

	- "self made"
	- "trivial"

THE FOLLOWING STEPS SHOULD BE DONE FOR THE TWO DIFFERENT DATA PROCESSING:

- 2 different embeddings:
	- "self made" glove
	- Stanford twitter 200d glove

- Classical Machine Learning:
	- Naive bayes
	- Logistic regression
	- Multilayer perceptron

THE FOLLOWING STEPS SHOULD BE DONE ONLY ON TRIVIAL DATA:

- 2 different embeddings:
	- "self made" glove
	- Stanford twitter 200d glove

- Advanced methods:
	- LSMT
	- CNN
