## Install Libraries and other Dependencies
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary

def loadText(file):
    with open(file, 'r') as text:
        doc = text.readlines()
    return doc

def preProcesslabel(label):
    # split into individual digits
    labels = [x.split() for x in label]
    # remove punctuations from digits
    labels = [[int(num) for num in line]
              for line in labels]
    return _normalze(labels)

def _normalze(label):
    labels = np.array(label)
    # normalize labels 
    return normalize(labels, norm='l1', axis=1)

# histogram and plots of text lengths
def textLengths(article, hist=False): 
    text_lengths = [len(line) for line in article]
    ax = plt.gca()        
    if hist:
        _hist(ax, text_lengths)
    else:
        _plot(ax, text_lengths)
        
def _plot(ax, text_lengths):
    # plots
    ax.plot(text_lengths, color='b')
    ax.set_xlabel('Sentence Index')
    ax.set_ylabel('Number of Words')
    ax.set_title('Plot of Number ' 
                 'of Words in a Text')
    
def _hist(ax, text_lengths):
    # histogram sentence size
    ax.hist(text_lengths, bins=10, color='r')
    ax.set_xlabel('Number of Words in a Text')
    ax.set_ylabel('Number of Texts')
    ax.set_title('Histogram of Number ' 
                 'of Words in a Text')

def removePunctuations(article):
    no_punctuation = [re.sub('[,\.!?]', '', text) 
                      for text in _to_lower_case(article)]
    return no_punctuation

# helper fuction to change texts to lower case
def _to_lower_case(article):
    lower_case = [text.lower() for text in article]
    return lower_case

## Tokenizing and Lemmatizing Data
def toToken(article):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenize = [tokenizer.tokenize(line) for line in article]  
    return tokenize


def lemmatize(article):
    article = _remove_numbers(article)
    article = _remove_single_characters(article)
    article = _remove_stopWords(article)
    lemmatizer = WordNetLemmatizer()
    lemmatize = [[lemmatizer.lemmatize(token) for token in line]
                  for line in article]
    lemmatize = _remove_single_characters(lemmatize)
    return lemmatize

# helper functions to clear numbers, sigle characters and stopwords
def _remove_numbers(article):
    no_numbers = [[token for token in line if not token.isnumeric()]
                  for line in article]
    return no_numbers

def _remove_single_characters(article):
    no_single_char = [[token for token in line if len(token) > 1]
                      for line in article]
    return no_single_char

def _remove_stopWords(article):     
    stop_words = stopwords.words('english')
    no_stopwords = [[token for token in line if token not in
                     stop_words] for line in article]
    return no_stopwords

## Vectorizing Data
def dictionary(article):
    # Create a dictionary representation of the acrticle
    dictionary = Dictionary(article)
    return dictionary

def wordToIndex(dictionary):
    # maps from word to the index of that word
    return dictionary.token2id
    
def indexToWord(dictionary):
    # maps from index to the word 
    word_to_index = wordToIndex(dictionary)
    if _is_empty(dictionary):
        index_to_word = {}
        for k in word_to_index:
            val = word_to_index[k]
            index_to_word[val] = k
    else:
        index_to_word = dictionary.id2token    
    return index_to_word

# helper funcion to check if dictionary is empty
def _is_empty(dictionary):
    return len(dictionary.id2token) == 0

def bagOfWords(article, dictionary):
    # Bag-of-words representation of the documents.
    bag_of_words = [dictionary.doc2bow(line)
              for line in article]
    return bag_of_words

## Saving and Loading Model 
def saveModel(model, file_name):
    return  model.save(file_name)

def loadModel(model, file_name):
    return model.load(file_name)
