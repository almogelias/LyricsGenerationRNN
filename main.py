import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pathlib
from skimage.measure import block_reduce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
import pretty_midi
from tqdm import tqdm_notebook as tqdm
import time
import datetime, math
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from tensorflow.keras.layers import Input, Embedding, Dense, GlobalAveragePooling1D, concatenate, LeakyReLU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
###
import glob

import tensorflow_hub as hub
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
import re
import pretty_midi


def remove_noise(lyrics):
    ret = lyrics.iloc[:, df_train.shape[1]-1].str.rstrip(r'&, ').str.extract(r'(.+)')
    return ret


def plot_word_freq(column):
    counts = Counter(column.str.cat(sep=' ').split(' '))
    labels, values = zip(*counts.items())

    print(f'Total number of words (raw): {len(labels)}')

    plt.plot(values)
    plt.show()
    return labels


def load_and_preprocess(csv_path, melody_dir=None, embedding=None, le=None, PAD=None, maxlen=6630):
    fix = ['concerned', 'sympathize', 'truthfulness', 'pondered', 'barbie', 'bimbo', 'dolly', 'hanky', 'hiya', 'ken', 'panky', "rock'n'roll", 'undress', 'commiserating', 'roses', 'surprises', 'windmill']
    ret = pd.read_csv(csv_path, usecols=[0, 1, 2], names=['artist', 'song', 'text'])

    # Get song path and add to dataframe
    if melody_dir is not None:
        artist_song = ret[['artist', 'song']].apply(lambda x: ' - '.join(x), axis=1)
        ret['song_path'] = list(map(
            lambda path: f'{melody_dir}/{path}',
            filter(
                lambda path: path.split('.')[0].replace('_', ' ').lower() in artist_song.values,
                os.listdir(melody_dir)
            )
        ))
        ret = ret.drop(['artist', 'song'], axis=1).reindex(columns=['song_path', 'text'])

    ret.text = remove_noise(ret)[0].str.split()

    if le is None:
        fit_data = ret.text.sum()+fix

        if PAD is not None:
            fit_data.append(PAD)
        le = LabelEncoder().fit(fit_data)

    ret['encoded'] = ret.text.apply(le.transform)

    if embedding is not None:
        ret['embedded'] = ret.text.apply(lambda text: get_weight_matrix(embedding, text))

    return ret, le


# Instead of choosing the word with highest probability,
# we choose the next word by sampling from the output probability distribution
def sample(vocab, preds):
    return np.random.choice(vocab, 1, p=preds)


def predict_step_by_step(model, intial_contexts, le, output_length=10):
    le_classes = list(le.classes_)
    print('####################')
    for intial_context in intial_contexts:
        print('--------------------------------')
        print(f'Generating lyrics for initial word {intial_context}')
        next_word = output_sentence = [intial_context]
        for i in range(output_length):
            context = next_word
            ds = tf.data.Dataset.from_tensor_slices((le.transform(context))).batch(batch_size=1)
            preds = model.predict(ds).squeeze()
            next_word = sample(le_classes, preds)
            output_sentence.append(next_word[0])
        print(' '.join(output_sentence).replace('& ', '&\n'))
        print(f"Done '{intial_context}'--------------------------------")


def load_embedding(filename=r'C:\Users\Anon\Downloads\Assignment 3\glove.6B.300d.txt'):
    with open(filename, 'r',encoding="utf8") as f:
        lines = f.readlines()

    # key is string word, value is numpy array for vector
    embedding = {parts[0]: np.asarray(parts[1:], dtype='float32') for parts in map(lambda line: line.split(), lines)}
    return embedding


# n*300 numpy array - Embedding layer - weight matrix
def get_weight_matrix(embedding, vocab):
    vocab_size = len(vocab)
    weight_matrix = np.zeros((vocab_size, 300))
    vectors = map(lambda word: embedding.get(word), vocab)
    for i, vector in enumerate(vectors):
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix


def create_tf_dataset(X, y, batch_size, buffer_size=1e4):
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size=batch_size).repeat()
    return ds


def create_and_train_LSTM_lyrics(
        train_data,
        log_dir='logs/fit',
        model=None,
        batch_size=4096, lr=1e-2, epochs=100, validation_split=0.2,
        patience=3, min_delta=.001,
        loss='sparse_categorical_crossentropy', optimizer=None, metrics=['accuracy'],
        prefix='', verbose=1,
        embedding_dim=300,
        PAD=' <PAD>', maxlen=6630,
        embedding=None, shuffle=True,
        random_state=2,
):
    if optimizer is None:
        optimizer = Adam(lr=lr)

    run_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
    run_name = f'{prefix}_{run_time}_lr{lr}_b{batch_size}'

    df_train, le_train = load_and_preprocess(csv_path=train_data, PAD=PAD, maxlen=maxlen)

    train_text = np.concatenate(df_train.encoded if PAD is None else df_train.encoded_padded)

    X = train_text[:-1]
    y = train_text[1:]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=random_state)

    ds_train = create_tf_dataset(X_train, y_train, batch_size=batch_size)
    ds_val = create_tf_dataset(X_val, y_val, batch_size=batch_size)

    vocab_size = len(le_train.classes_)

    if embedding is not None:
        word_vectors = get_weight_matrix(embedding, le_train.classes_)
        embedding_layer = Embedding(
            vocab_size,
            embedding_dim,
            weights=[word_vectors],
            trainable=False,
        )
    else:
        embedding_layer = Embedding(
            vocab_size, embedding_dim,
        )

    # define model
    if model is None:
        model = Sequential([
            embedding_layer,
            LSTM(embedding_dim),
            Dense(vocab_size, activation='softmax')
        ])
        print(model.summary())

    callbacks = [
        EarlyStopping(patience=patience, min_delta=min_delta, verbose=verbose),
        ModelCheckpoint(f'{run_name}.h5', verbose=0, save_best_only=True, save_weights_only=True)
    ]

    if tf.__version__[0] == '2':
        callbacks.append(TensorBoard(log_dir=f'{log_dir}/{run_name}'))

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)

    history = model.fit(
        ds_train,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
        steps_per_epoch=math.ceil(len(X_train) / batch_size),
        #       batch_size=batch_size,
        validation_data=ds_val,
        validation_steps=math.ceil(len(X_val) / batch_size),
        #       validation_split=validation_split
    )
    return model, le_train, callbacks


def model_eval_LSTM_lyrics(
        test_data,
        model,
        le,
        callbacks=None,
        batch_size=4096,
        verbose=1,
        PAD=' <PAD>', maxlen=6630,
):
    df_test, le_test = load_and_preprocess(csv_path=test_data, PAD=PAD, maxlen=maxlen, le=le)

    test_text = np.concatenate(df_test.encoded if PAD is None else df_test.encoded_padded)

    ds_test = create_tf_dataset(test_text[:-1], test_text[1:], batch_size=batch_size)

    model.evaluate(
        ds_test,
        callbacks=callbacks,
        verbose=verbose,
        steps=math.ceil((len(test_text) - 1) / batch_size),
    )
if __name__ == '__main__':
    #Analyze database

    repository_path = os.path.join(r'C:\Users\Anon\Downloads', 'Assignment 3')
    train_melody_path = os.path.join(repository_path, 'midi_files', 'train')
    test_melody_path = os.path.join(repository_path, 'midi_files', 'test')
    train_data = os.path.join(repository_path, 'lyrics_train_set.csv')
    test_data = os.path.join(repository_path, 'lyrics_test_set.csv')


    df_train = pd.read_csv(train_data, usecols=[0, 1, 2], names=['artist', 'song', 'text'])

    df_train.text = remove_noise(df_train)

    vocab = plot_word_freq(df_train.text)

    counts = Counter(df_train.text.str.cat(sep=' ').split(' '))
    labels, values = zip(*counts.items())
    print(len(labels))
    print(df_train.text.str.len().max())
    df_train.head()

    df_tmp, le_tmp = load_and_preprocess(csv_path=train_data, PAD=None, maxlen=6630)
    print('Loading Embeddings')
    word2vec_embedding_original = load_embedding()
    print('Finish Loading Embeddings')

    batch_size = 4096 * 4

    model_pre_train, le_pre_train, cb_pre_train = create_and_train_LSTM_lyrics(
        train_data,
        batch_size=batch_size, lr=1e-3, validation_split=0.2,
        epochs=100,
        prefix='pre_train', verbose=1,
        PAD=None,
        embedding=word2vec_embedding_original
    )
    model_pre_train.summary()
