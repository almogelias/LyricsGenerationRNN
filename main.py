
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
import numpy as np
import re
import pretty_midi
import pandas as pd




def wordList(lyrics_train_set,word2vec_file):
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

    embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen,
                                weights=[emb_matrix], trainable=False)

    return embedding_layer

def read_glove_vector(glove_vec):
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
      w_line = line.split()
      curr_word = w_line[0]
      word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)

  return word_to_vec_map

if __name__ == '__main__':

    lyrics_train_set = r'C:\Users\Anon\Downloads\Assignment 3\lyrics_train_set.csv'
    word2vec_file=r'C:\Users\Anon\Downloads\Assignment 3\glove.6B.300d.txt'
    embedding_layer=wordList(lyrics_train_set,word2vec_file)

    midi_pretty_format = pretty_midi.PrettyMIDI(r'C:\Users\Anon\Downloads\Assignment 3\midi_files\adele_-_someone_like_you.mid')
    piano_midi = midi_pretty_format.instruments[0]  # Get the piano channels
    piano_roll = piano_midi.get_piano_roll(fs=30)
    # Load Pretrained Word2Vec
    #embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
    hub_url = r"C:\Users\Anon\Downloads\Assignment 3\Wiki-words-250_2"
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
