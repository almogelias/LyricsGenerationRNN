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
###
import glob

import tensorflow_hub as hub
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
import re
import pretty_midi


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
    ret = ret.str.replace(r'[\[\]()\{\}:;#\*"ã¤¼©¦­]', '')

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
    with open(filename, 'r') as f:
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
    ret = pd.read_csv(train_data, usecols=[0, 1, 2], names=['artist', 'song', 'text'])

    # Get song path and add to dataframe
    artist_song = ret[['artist', 'song']].apply(lambda x: ' - '.join(x), axis=1)
    ret['song_path'] = list(map(
        lambda path: f'{melody_dir}/{path}',
        filter(
            lambda path: path.split('.')[0].replace('_', ' ').lower() in artist_song.values,
            os.listdir(melody_dir)
        )
    ))
    ret = ret.drop(['artist', 'song'], axis=1).reindex(columns=['song_path', 'text'])

    ret.text = remove_noise_char(ret.text).str.split()

    if embedding is not None:
        ret['text_embedded'] = ret.text.apply(lambda text: get_weight_matrix(embedding, text))

    if le is None:
        le = LabelEncoder().fit(ret.text.sum())

    ret['text_encoded'] = ret.text.apply(le.transform)
    return ret, le


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

    repository_folder = r'C:\Users\Anon\Downloads\Assignment 3'
    word2vec_file=r'C:\Users\Anon\Downloads\Assignment 3\glove.6B.300d.txt'
    midi_files_folder = r'C:\Users\Anon\Downloads\Assignment 3\midi_files'

    train_melody_path = os.path.join(repository_folder, 'data', 'midi_files', 'train')
    test_melody_path = os.path.join(repository_folder, 'data', 'midi_files', 'test')
    train_data = os.path.join(repository_folder, 'lyrics_train_set.csv')
    test_data = os.path.join(repository_folder, 'lyrics_test_set.csv')

    df_training = words_df(train_data)
    labels = plot_words_frequency(df_training[2])
    raw_embedding = load_embedding(word2vec_file)

    #PreProcess DataSet

    print(len(labels))
    print(df_train.text.str.len().max())
    df_train.head()
