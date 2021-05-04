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

def compare_train_midifolder(train_data,train_melody_path):
    '''This function is to compare between train csv file and midi collection'''
    midi_songs_name_list=[]
    train_data_songs=[]
    list_of_melodydir=os.listdir(train_melody_path)
    for midi in list_of_melodydir:
        midi_songs_name_list.append(midi.replace('_', ' ').split('.')[0].split(' - ')[1].lower())
    ret = pd.read_csv(train_data, usecols=[1],header=None)
    # dropping ALL duplicte values
    ret.drop_duplicates(keep=False, inplace=True)

    train_data_songs=ret.values[:,0].tolist()
    diffrence=np.setdiff1d(train_data_songs, midi_songs_name_list).tolist()
    duplications = set([x for x in midi_songs_name_list if midi_songs_name_list.count(x) > 1])
    for dup in duplications:
        diffrence.append(dup)

    if diffrence:
        return False
    else:
        return True

def remove_noise(text):
    # Replace '' with " (later remove it)
    ret = text.str.replace(r"''", '"')

    # Replace ` with '
    ret = ret.str.replace(r'`', "'")

    # Replace / with - (appears in 24/7)
    ret = ret.str.replace(r'/', '-')

    # Replace three ? with one
    ret = ret.str.replace('\?\?\?', '?')

    # Replace all brackets, :;#", and weird chars
    ret = ret.str.replace(r'[\[\]()\{\}:;#\*"ã¤Ÿ¼©¦­]', '')

    # Remove certain words
    for word in ['chorus', '\-\-\-', '\-\-']:
        ret = ret.str.replace(word, '')

    # Replace these words with space
    for word in ['\.\.\.\.\.', '\.\.\.\.', '\.\.\.', '\.\.', " '"]:
        ret = ret.str.replace(word, ' ')

    # Remove duplicate spaces
    ret = ret.apply(lambda s: ' '.join(s.split()))
    return ret


def plot_words_frequency(lines):
    counters = Counter(lines.str.cat(sep=' ').split(' '))
    labels, values = zip(*counters.items())
    print(f'Total # words: '.format(len(labels)))
    plt.plot(values)
    plt.show()
    return labels


def words_df(lyrics_train_set):
    df = pd.read_csv(lyrics_train_set, header=None)
    df.loc[:, 2] = df[df.columns[2:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    df.drop(df.iloc[:, 3:], inplace=True, axis=1)
    df[2]= remove_noise(df[2])
    return df


def get_piano_rolls_songs(melody_path, partial=None):
    melody_paths = list(map(str, pathlib.Path(melody_path).glob('*')))
    if partial is not None:
        melody_paths = melody_paths[:partial]

    return melody_paths


# Load the pre-trained Glove Word2Vec embeddings
def load_embedding(filename='glove_pretrained/glove.6B.300d.txt'):
    """
    input:
    @filename: a .txt file of the pre-trained word2vec embedding

    output:
    @embedding: a dictionary of word vectors. Keys are distinct words and values are the word vectors
    """
    with open(filename, 'r',encoding="utf8") as f:
        lines = f.readlines()

    # key is string word, value is numpy array for vector
    embedding = {parts[0]: np.asarray(parts[1:], dtype='float32') for parts in map(lambda line: line.split(), lines)}
    return embedding


# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    """
    input:
    @embedding: dictionary of word vectors (all words in the pre-trained model)
    @vocab: vocabulary of the lyrics dataset

    output:
    @weight_matrix: a n*300 numpy array, where n is the total number of words in the vocabulary.
                    each row of this matrix is a 300 dimension word vector of the ith word in vocab.
    """

    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab)
    #     print(f'There are {vocab_size} words in this song.')
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, 300))
    # step vocab, store vectors using the Tokenizer's integer mapping
    vectors = map(lambda word: embedding.get(word), vocab)
    for i, vector in enumerate(vectors):
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix


def load_and_preprocess(csv_path, melody_dir, embedding=None, le=None):
    res_df = pd.read_csv(train_data, usecols=[0, 1, 2], names=['artist', 'song', 'text'])
    res_df.drop_duplicates(keep=False, inplace=True)
    # Get song path and add to dataframe
    artist_song = res_df[['artist', 'song']].apply(lambda x: ' - '.join(x), axis=1)

    files=[]
    list_songs_path = list(map(
        lambda path: f'{path}',
        filter(
            lambda path: path.split('.')[0].replace('_', ' ').lower() in artist_song.values,
        os.listdir(melody_dir)
        )
    ))
    for path in list_songs_path:
        files.append(path.split('.')[0].replace('_', ' ').lower())
    diffrence= set(artist_song.values.tolist())^set(files)
    if(diffrence):
        print('Please fix diffrences between csv and midi files: {}'.format(diffrence))

    else:
        res_df=res_df.sort_values('artist')
        list_to_add_ret=[]
        for index, row in res_df.iterrows():
            song_file=row['artist']+' - '+row['song']+'.mid'
            for name in list_songs_path:
                if (name.lower().replace('_',' '))==(song_file):
                    list_to_add_ret.append(name)
        res_df['song_path']=list_to_add_ret
        res_df = res_df.drop(['artist', 'song'], axis=1).reindex(columns=['song_path', 'text'])

        res_df.text = remove_noise(res_df.text).str.split()

        if embedding is not None:
            res_df['text_embedded'] = res_df.text.apply(lambda text: get_weight_matrix(embedding, text))

        if le is None:
            label_e = LabelEncoder().fit(res_df.text.sum())

        res_df['text_encoded'] = res_df.text.apply(label_e.transform)
        return res_df, label_e


def down_sample_piano_roll(piano_roll, block_size=(4, 4)):
    return block_reduce(
        block_reduce(piano_roll, block_size=block_size, func=np.mean),
        block_size=(4, 4), func=np.mean
    )


def load_and_preprocess_melody(
        melody_path,
        normalize=None,
        th_len=4000000,  # 7696768,
        block_size=(4, 4),
):
    try:
        ret = pretty_midi.PrettyMIDI(melody_path).get_piano_roll()
        if block_size is not None:
            ret = down_sample_piano_roll(ret, block_size=block_size)
        #     ret = ret.sum(axis=0)
        ret = ret.flatten()
        # print('Piano roll Loaded.')
        if th_len is not None:
            if len(ret) > th_len:
                # print('Piano roll too long, trimming...')
                ret = ret[:th_len]
            else:
                # print('Piano roll too short, padding...')
                ret = tf.keras.preprocessing.sequence.pad_sequences(
                    [ret],
                    value=0,
                    padding='post',
                    maxlen=th_len)[0]
        if normalize is not None:
            ret = normalize(ret, norm=normalize, axis=1)
        return ret
    except:
        print('EXCEPT', melody_path)
        return None


def input_generator(
        df,
        batch_size,
        timesteps=1,
        normalize=None,
        th_len=4000000,  # 7696768,
        block_size=(4, 4),
):
    batch_id = 0

    #   print(f'Batch number {batch_id}')
    def helper():
        for song_id, row in tqdm(df.iterrows(), desc='song loop', total=df.shape[0]):
            #       print(batch_id, row['song_path'])
            pm = load_and_preprocess_melody(row['song_path'], normalize=None, th_len=th_len, block_size=block_size)
            for curr_word, next_word in zip(row['text_embedded'][:-1], row['text_encoded'][1:]):
                yield (curr_word, pm), next_word

    generator = helper()
    while True:
        #     start = time.time()
        curr_word_len = df.iloc[0]['text_embedded'][0].shape[0]
        curr_word_batch = np.zeros((batch_size * timesteps, curr_word_len))
        next_word_batch = np.zeros((batch_size * timesteps, 1))
        pm_batch = np.zeros((batch_size * timesteps, th_len))

        for i, ((curr_word, pm), next_word) in zip(range(batch_size), generator):
            curr_word_batch[i] = curr_word
            next_word_batch[i] = next_word
            pm_batch[i] = pm

        curr_word_batch = curr_word_batch.reshape(batch_size, timesteps, curr_word_len)
        next_word_batch = next_word_batch.reshape(batch_size, timesteps)
        pm_batch = pm_batch.reshape(batch_size, timesteps, th_len)
        #     end = time.time()
        #     print(end - start)

        yield (curr_word_batch, pm_batch), next_word_batch
        batch_id += 1



def sample(vocab, preds):
  # helper function to sample an index from a probability array
  ret = np.random.choice(vocab, 1, p = preds)
  return ret

def predict_step_by_step(model, song_path, initial_word, le, embedding, th_len=12500, block_size=(4,4), output_length=100):
  pm = load_and_preprocess_melody(song_path, th_len=th_len, block_size=block_size).reshape(1, 1, th_len)
  next_word = output_sentence = [initial_word]
  for i in range(output_length):
    curr_word = next_word
    preds = model.predict((get_weight_matrix(embedding, curr_word).reshape(1, 1, 300), pm)).squeeze()
    next_word = sample(list(le.classes_), preds)
    #next_word = np.argmax(preds, axis=1)
    output_sentence.append(next_word[0])
  print(' '.join(output_sentence).replace('& ', '&\n'))


def create_and_train_LSTM(
        train_data,
        melody_dir,
        generator,
        model=None,
        normalize=None,
        timesteps=1,
        batch_size=4096, lr=1e-2, epochs=100, validation_split=0.2,
        patience=3, min_delta=.001,
        loss='sparse_categorical_crossentropy', optimizer=None, metrics=['accuracy'],
        prefix='', verbose=2,
        embedding_dim=300,
        lstm_units=300,
        embedding=None,
        th_len=4000000,  # 7696768,
        block_size=(4, 4),
        random_state=2
):
    if embedding is None:
        embedding = load_embedding()
    if optimizer is None:
        optimizer = Adam(lr=lr)

    run_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
    run_name = f'{prefix}_{run_time}_lr{lr}_b{batch_size}'

    df, label_e_train = load_and_preprocess(train_data, melody_dir, embedding)

    df_train, df_val = train_test_split(df, test_size=validation_split, random_state=random_state)
    print(f'Train length: {df_train.shape[0]}', f'Train words: {len(df_train.text.sum())}',
          f'Validation length: {df_val.shape[0]}', f'Validation words: {len(df_val.text.sum())}')

    vocab_size = len(label_e_train.classes_)

    # define model
    if model is None:
        curr_words_input = Input(shape=(timesteps, embedding_dim), name='curr_words')
        melody_input = Input(shape=(timesteps, th_len), name='melody')

        combined = concatenate([curr_words_input, melody_input], name='combined')
        x = LSTM(units=lstm_units, name='LSTM',
                 )(combined)
        x = Dense(vocab_size, activation='softmax', name='Dense')(x)

        model = Model(inputs=[curr_words_input, melody_input], outputs=x)

        print(model.summary())

    callbacks = [
        EarlyStopping(patience=patience, verbose=verbose, min_delta=min_delta),
        ModelCheckpoint(f'{run_name}.h5', verbose=0, save_best_only=True, save_weights_only=True)
    ]

    if tf.__version__[0] == '2':
        log_dir = f'logs/fit/{run_name}'
        callbacks.append(TensorBoard(log_dir=log_dir))

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)

    history = model.fit_generator(
        generator=generator(df_train, timesteps=timesteps, normalize=normalize, batch_size=batch_size, th_len=th_len,
                            block_size=block_size),
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
        #       batch_size=batch_size,
        steps_per_epoch=math.ceil(len(df_train.text.sum()) / batch_size),
        validation_data=generator(df_val, batch_size=batch_size, th_len=th_len, block_size=block_size),
        validation_steps=math.ceil(len(df_val.text.sum()) / batch_size)
    )
    return model, le_train



def wordListOld(lyrics_train_set,word2vec_file):
    word_dict = {}
    word_list = []
    df = pd.read_csv(lyrics_train_set, header=None)
    df.loc[:, 2] = df[df.columns[2:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    df.drop(df.iloc[:, 3:], inplace=True, axis=1)
    for idx, line in enumerate(df.loc[:][2]):
        string_no_punctuation = re.sub("[^\w\s]", "", line)
        word_dict[idx] = string_no_punctuation.lower()
        word_list.append(word_dict[idx])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_list)
    words_to_index = tokenizer.word_index

    word_to_vec_map = read_glove_vector(word2vec_file)
    maxLen = 300

    vocab_len = len(words_to_index)
    embed_vector_len = word_to_vec_map['test'].shape[0]

    emb_matrix = np.zeros((vocab_len, embed_vector_len))

    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index-1, :] = embedding_vector

    #embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen,
    #                            weights=[emb_matrix], trainable=False)

    return emb_matrix

def read_glove_vector(glove_vec):
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
      w_line = line.split()
      curr_word = w_line[0]
      word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)

  return word_to_vec_map



def wordList(midi_files_folder):

    for file_path in glob.glob(midi_files_folder+'\\*.mid'):

        try:
            midi_pretty_format = pretty_midi.PrettyMIDI(file_path)
            print('There are {} time signature changes'.format(len(midi_pretty_format.time_signature_changes)))
            print('There are {} instruments'.format(len(midi_pretty_format.instruments)))
            print('Instrument 3 has {} notes'.format(len(midi_pretty_format.instruments[0].notes)))
            print('Instrument 4 has {} pitch bends'.format(len(midi_pretty_format.instruments[4].pitch_bends)))
            print('Instrument 5 has {} control changes'.format(len(midi_pretty_format.instruments[5].control_changes)))
            print("Instruments")
            print(len(midi_pretty_format.instruments))
            print("Matrix")
            print(midi_pretty_format.get_pitch_class_transition_matrix().shape)
            print("GetChroma")
            print(midi_pretty_format.get_chroma().shape)
            print("PianoRoll")
            print(midi_pretty_format.get_piano_roll().shape)
        except:
            print(midi_pretty_format)

    return True


if __name__ == '__main__':

    tf.random.set_seed(2)

    repository_folder = r'C:\Users\xadmin\Downloads\Assignment 3\Assignment 3'
    word2vec_file=r'C:\Users\xadmin\Downloads\Assignment 3\Assignment 3\glove.6B.300d.txt'
    midi_files_folder = r'C:\Users\xadmin\Downloads\Assignment 3\Assignment 3\midi_files'


    train_melody_path = os.path.join(repository_folder, 'midi_files', 'train')
    test_melody_path = os.path.join(repository_folder, 'midi_files', 'test')
    train_data = os.path.join(repository_folder, 'lyrics_train_set.csv')
    test_data = os.path.join(repository_folder, 'lyrics_test_set.csv')

    #check_train_midifolder = compare_train_midifolder(train_data,train_melody_path)
    df_training = words_df(train_data)
    labels = plot_words_frequency(df_training[2])
    raw_embedding = load_embedding(word2vec_file)
    th_len = 12500  # 4000000
    block_size = (4, 4)
    normalize = 'l1'
    lstm_units = 128
    timesteps = 5

    model_down_sample, le_down_sample = create_and_train_LSTM(
        train_data,
        melody_dir=train_melody_path,
        generator=input_generator,
        normalize=normalize,
        timesteps=timesteps,
        batch_size=1024, lr=1e-2, patience=3, validation_split=0.2,
        prefix='down_sample',
        embedding=raw_embedding,
        lstm_units=lstm_units,
        th_len=th_len,
        block_size=block_size,
    )

    #Prediction
    predict_step_by_step(model_down_sample, '/content/AutoLyrics/data/midi_files/train/ABBA_-_Money_Money_Money.mid',
                         'i', le_down_sample, raw_embedding, th_len=th_len, block_size=block_size, output_length=100)

    train_data_generator = load_and_preprocess_melody(melody_path, contexts, melodies, next_words, raw_embedding)

    model.fit_generator(train_data_generator, steps_per_epoch=len(contexts), epochs=10)
    count = 1
    for i in train_data_generator:
        print(count)
        print(i[0].shape)
        count += 1
    #PreProcess DataSet

    print(len(labels))
    print(df_train.text.str.len().max())
    df_train.head()
