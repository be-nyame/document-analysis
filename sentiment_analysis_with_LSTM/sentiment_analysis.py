# importing necessary libraries

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras import layers
from keras.models import load_model 
from scipy.stats import pearsonr
from math import sqrt
import re

from utils import *   # contains data preprocessing 
                      # and other related
                      # functions

## Model Evaluation with Acc@1 and
## AP: average Pearson correlation coefficien

# Evaluation with Acc@1                      
def acc1(y, y_hat):
    return sum(_index(y, y_hat)
              ) / len(y_hat)

def _index(y, y_hat):
    """
      helper function to get index of highest predicted 
      and highest label probalilities for each
      observation. Sets 1 if they are equal and 0
      otherwise
    """
    acc_1 = []
    for idx in range(len(y_hat)):
        if np.argmax(y_hat[idx]) == np.argmax(
            y[idx]):
            i_d = 1
        else:
            i_d = 0
        acc_1.append(i_d)
    return acc_1

# Evaluation with AP
def avgPearsonCorrelationCoefficient(y, y_hat):
    return sum(_pearson_correlation_coefficient(y, y_hat)
              ) / len(y_hat)

def _pearson_correlation_coefficient(y, y_hat):
    r = []
    for idx in range(len(y_hat)):
        # pearson correlation coefficients with
        # pearsonr from scipy library
        value = pearsonr(y[idx], y_hat[idx])
        pear_coeff, p_value = value
        r.append(pear_coeff)
    return r

def saveEmotions(file_name, prediction):
    # saving predictions
    predicted = _label_to_string(prediction)
    with open(file_name, 'w') as text:
        for i in predicted:
            text.write("%s\n" % i)

def _label_to_string(prediction):
    #stringify each label 
    predicted = [' '.join(str(round(num,3)) for num in c) 
                 for c in prediction.tolist()]
    return predicted

def plotGraphs(history, metric):
    # plotting training and validation accuracy and losses
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    if metric == 'acc':
        plt.title('Training and validation accuracy')
    else:
        plt.title('Training and validation loss')
    plt.show()

if __name__ == "__main__":
    import warnings
    # ignore warnings
    warnings.filterwarnings('ignore')

    train_articles = loadText('train.txt')
    train_doc = train_articles
    print('first five lines of training articles',
          train_doc[:5])

    print("size of train set: {}".format(len(train_doc)))

    test_articles = loadText('test.txt')
    test_doc = test_articles
    print('first five lines of testing articles',
          test_doc[:5])

    print("size of test set: {}".format(len(test_doc)))

    label = loadText('label.txt')

    print('first five lines of labels',
          label[:5])

    print("size of label set: {}".format(len(label)))

    labels = preProcesslabel(label)

    print('first five lines of preprocessed labels',
          labels[:5])

    print('Train set')
    plt.show(textLengths(train_doc))
    plt.show(textLengths(train_doc, hist=True))

    print('Test Set')
    plt.show(textLengths(test_doc))
    plt.show(textLengths(test_doc, hist=True))

    train_doc = removePunctuations(train_doc)
    test_doc = removePunctuations(test_doc)

    # splitting data into 75.4% train and 24.6% validation data
    val_size = 0.246
    texts_train, texts_val, label_train, label_val = train_test_split(
        train_doc, labels, test_size=val_size, random_state=42) 
    print("size of training set:", len(texts_train))
    print("size of validation set:", len(texts_val))
    print("size of label training set:", len(label_train))
    print("size of label validation set:", len(label_val))

    # splitting texts into tokens
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts_train)
    # changing tokens into into sequence of numbers
    train_data_seq = tokenizer.texts_to_sequences(texts_train)
    val_data_seq = tokenizer.texts_to_sequences(texts_val)
    test_data_seq = tokenizer.texts_to_sequences(test_doc)

    # padding sequence of numbers with sequence lenght of 50
    seq_len = 50
    train_texts = sequence.pad_sequences(train_data_seq, maxlen=seq_len)
    val_texts = sequence.pad_sequences(val_data_seq, maxlen=seq_len)
    test = sequence.pad_sequences(val_data_seq, maxlen=seq_len)

    # padding sequence of numbers with sequence lenght of 50
    seq_len = 50
    train_texts = sequence.pad_sequences(train_data_seq, maxlen=seq_len)
    val_texts = sequence.pad_sequences(val_data_seq, maxlen=seq_len)
    test = sequence.pad_sequences(val_data_seq, maxlen=seq_len)

    print('train data shape:', train_texts.shape)
    print('validation data shape:', val_texts.shape)
    print('test data shape:', test.shape)

    # one-hot encoding labels into size of class
    num_classes = 6
    train_label = to_categorical(label_train, num_classes=num_classes)
    val_label = to_categorical(label_val, num_classes=num_classes)

    # defining model architecture
    out_dim = 128
    n_units = 32
    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential()
    model.add(Embedding(vocab_size, out_dim, input_length=seq_len))
    model.add(LSTM(n_units))
    model.add(Dense(n_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) 

    # displaying model achitecture
    model.summary()

    # defining model evaluation parameters
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy']) 

    # model training for 10 epochs
    history = model.fit(train_texts, label_train,
                        batch_size=32, epochs=10,
                        validation_data=(val_texts, label_val))

    # saving learned parameters

    file_name = 'sentiment_analysis_01.h5'
    saveModel(model, file_name)

    plotGraphs(history, 'acc')
    plotGraphs(history, 'loss')

    ## Increasing Model Architecture and Applying Regularization

    # new model architecture with extra LSTM layer
    # and addition of dropouts
    out_dim = 128
    n_units = 32
    vocab_size = vocab_size
    model = Sequential()
    model.add(Embedding(vocab_size, out_dim, input_length=seq_len))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(LSTM(n_units))
    model.add(layers.Dropout(0.2))
    model.add(Dense(n_units, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))  

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics= ['accuracy'])


    # training model for 10 epochs
    history = model.fit(train_texts, label_train,
                       batch_size=128, epochs=10,
                       validation_data=(val_texts, label_val))
    file_name = 'sentiment_analysis_02.h5'
    saveModel(model, file_name)

    # plotting training and validation accuracy and losses
    plotGraphs(history, 'acc')
    plotGraphs(history, 'loss')

    ## Model Evaluation with Acc@1 and
    ## AP: average Pearson correlation coefficien
    # loading learned parameters from model
    model = load_model(file_name)

    # get predicted values
    prediction = model.predict(test)

    y = labels
    y_hat = prediction

    # Evaluation with Acc@1
    print('Model Acc@1 =', acc1(y, y_hat))

    # Evaluation with AP
    print('Average Pearson Correlation Coefficient =',
          avgPearsonCorrelationCoefficient(y, y_hat))

    # Saving Model Predictions as Text Document
    saveEmotions('emotion.txt', prediction)


























    
