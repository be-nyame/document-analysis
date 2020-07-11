## Install Libraries and other Dependencies

from matplotlib import pyplot as plt
from gensim.models import LdaModel
import logging
import warnings
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim
import pandas as pd
import seaborn as sns

from utils import *   # contains data preprocessing 
                      # and other related
                      # functions

# concatenate train and test sets
def article(train_doc, test_doc):
    return (train_doc + test_doc)


def printTopics(model, corpus, proba=False):
    if proba:
        return pprint(model.top_topics(corpus))
    else:
        return pprint(model.print_topics())


## Model Evaluation
# calculates the C_v topic coherence using Palmetto
# https://palmetto.demos.dice-research.org/

def topFiveWords(model):
    top_five = []
    for i in range(num_topics):
        # create dataframe of model topics
        data_frame = _words_data_frame(model, i)
        top_five.append([i for i in data_frame['term'].head(5)])
    return top_five

def showTopFive(top_five_words, prob, coherence=False):
    indx = 0
    topic_num = 1
    for words in top_five_words:
        if not coherence:
            print("top five words of topic{}: {}".format(
                topic_num, words))
        else:
            print("topic{}-{}: {}".format(
                topic_num, words, prob[indx]))
        indx +1
        topic_num += 1

def _words_data_frame(model, index, plot=False):
    data_frame = pd.DataFrame(model.show_topic(index), 
                                 columns=
                                 ['term','prob'])
    if plot:
        data_frame = data_frame.set_index('term')
    else:
        data_frame = data_frame
    return data_frame


# C_v topic coherence with Palmetto
# obtained results from: 
# https://palmetto.demos.dice-research.org/

prob = [0.4603991699723064, 0.46531399665204853, 
        0.40958670895514926, 0.4534868002432657,
        0.45794657132243904, 0.40874119413485327,
        0.40625418388639567, 0.41724468745735105,
        0.4580563176741944, 0.45117204379041115]


# average C_v topic coherence of 10 topics
def average_coherence(coherence):
    return sum(coherence)/num_topics

## Model Visualization

def words_importance(model):
    fig = plt.figure(figsize=(15,39))
    for i in range(num_topics):
        # create dataframe of model topics
        data_frame = _words_data_frame(model, i, plot=True)
        plt.subplot(5,2,i+1)
        plt.title('topic '+str(i+1))
        sns.barplot(x='prob', y=data_frame.index, data=data_frame,
                   label='Cities', palette='Reds_d')
        plt.xlabel('probability')


def viewModel(model, corpus, dictionary):
    return pyLDAvis.gensim.prepare(
        model, corpus, dictionary)


if __name__ == "__main__":
        
    # ignore warnings
    warnings.filterwarnings('ignore')

    # get updates
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', 
        level=logging.INFO)

    ## Load and Prepare the Dataset
    
    train_doc = loadText('train.txt')
    test_doc = loadText('test.txt')
    print("size of train set: {}".format(
    len(train_doc)))
    print("first five articles in train set:",
          train_doc[:5])

    print("size of test set: {}".format(
    len(train_doc)))
    print("first five articles in test set:",
          train_doc[:5])

    # Concatenate train and test sets for articles
    article = article(train_doc, test_doc)
    print("size of article: {}".format(
        len(article)))

    # histogram and plots of text lengths
    plt.show(textLengths(article))
    plt.show(textLengths(article, hist=True))

    ## Preprocessing Data for Training

    # Data Cleaning
    article = removePunctuations(article)


    # Tokenizing and Lemmatizing Data
    article = toToken(article)
    article = lemmatize(article)
    print("tokenized and lemmatized texts: {}".format(
        article[:5]))

    # Vectorizing Data
    dictionary = dictionary(article)

    index_to_word = indexToWord(dictionary)
    corpus = bagOfWords(article, dictionary)
    
    print('First Five Corpus:', corpus[0][:5])
    print('Number of unique tokens: %d' % len(
        dictionary))
    print('Number of documents: %d' % len(corpus))
    
    ## Model Training

    # set training parameters.
    num_topics = 10
    passes = 20
    iterations = 2 #400

    # train model
    model = LdaModel(corpus=corpus, id2word=index_to_word, 
                         num_topics=num_topics, passes=passes,
                         iterations=iterations)

    ## Saving and Loading Model
    file_name = 'LDA.model'
    saveModel(model, file_name)

    model_LDA = loadModel(model, file_name)

    # display topics
    printTopics(model=model_LDA, corpus=corpus)

    # top 5 words of each topic 
    top_five_words = topFiveWords(model_LDA)
    showTopFive(top_five_words, prob)

    print ("average C_v topic coherence of the top" 
       "10 topics: {}".format(
           round(average_coherence(prob),3)))
    
    showTopFive(top_five_words, prob, coherence=True)

    ## Model Visualization in matplotlib
    ## and pyLDAvis
    
    plt.show(words_importance(model_LDA))

    view_model = viewModel(model_LDA, corpus, dictionary)
    pyLDAvis.show(view_model) 

