
import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import re
import pretty_midi
import pandas as pd




def wordList(lyrics_train_set):
    word_list = {}
    df = pd.read_csv(lyrics_train_set, header=None)
    df.loc[:, 2] = df[df.columns[2:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    df.drop(df.iloc[:, 3:], inplace=True, axis=1)
    for idx, line in enumerate(df.loc[:][2]):
        string_no_punctuation = re.sub("[^\w\s]", "", line)
        word_list[idx] = string_no_punctuation.split()
    return word_list


def get_max_length(df):
    """
    get max token counts from train data,
    so we use this number as fixed length input to RNN cell
    """
    max_length = 0
    for row in df['review']:
        if len(row.split(" ")) > max_length:
            max_length = len(row.split(" "))
    return max_length


def get_word2vec_enc(reviews):
    """
    get word2vec value for each word in sentence.
    concatenate word in numpy array, so we can use it as RNN input
    """
    encoded_reviews = []
    for review in reviews:
        tokens = review.split(" ")
        word2vec_embedding = embed(tokens)
        encoded_reviews.append(word2vec_embedding)
    return encoded_reviews


def get_padded_encoded_reviews(encoded_reviews):
    """
    for short sentences, we prepend zero padding so all input to RNN has same length
    """
    padded_reviews_encoding = []
    for enc_review in encoded_reviews:
        zero_padding_cnt = max_length - enc_review.shape[0]
        pad = np.zeros((1, 250))
        for i in range(zero_padding_cnt):
            enc_review = np.concatenate((pad, enc_review), axis=0)
        padded_reviews_encoding.append(enc_review)
    return padded_reviews_encoding


def sentiment_encode(sentiment):
    """
    return one hot encoding for Y value
    """
    if sentiment == 'positive':
        return [1, 0]
    else:
        return [0, 1]


def preprocess(df):
    """
    encode text value to numeric value
    """
    # encode words into word2vec
    reviews = df['review'].tolist()

    encoded_reviews = get_word2vec_enc(reviews)
    padded_encoded_reviews = get_padded_encoded_reviews(encoded_reviews)
    # encoded sentiment
    sentiments = df['sentiment'].tolist()
    encoded_sentiment = [sentiment_encode(sentiment) for sentiment in sentiments]
    X = np.array(padded_encoded_reviews)
    Y = np.array(encoded_sentiment)
    return X, Y

if __name__ == '__main__':

    lyrics_train_set = r'C:\Users\xadmin\Downloads\Assignment 3\Assignment 3\lyrics_train_set.csv'
    word_list=wordList(lyrics_train_set)

    # Load Pretrained Word2Vec
    #embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
    hub_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
    embed = hub.KerasLayer(hub_url)
    embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
    embed(["friend"])
    movie_reviews_train = [
             {'review': 'this is the best movie', 'sentiment': 'positive'},
             {'review': 'i recommend you watch this movie', 'sentiment': 'positive'},
             {'review': 'it was waste of money and time', 'sentiment': 'negative'},
             {'review': 'the worst movie ever', 'sentiment': 'negative'}
        ]
    df = pd.DataFrame(movie_reviews_train)

    # max_length is used for max sequence of input
    max_length = get_max_length(df)

    train_X, train_Y = preprocess(df)