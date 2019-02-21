
"""
Input: Texts contains multiple sentence
Output: 1 line sentence
Area: Text Summarization, Chatbot, Q&A, Machine Translation
Goal: From unstructuered text to Embedding layer
"""

"""
1. Preparation: from text to sentence
Non-alphanumeric data removing: number, symbol, emoji, HTML tag…
Lowercase and Miss-spelling normalization
Tokenization: from sentence to word
Stop Words removing
Stemming and Lemmatization
Morphological Analysis(POS) or Named Entity Recognition
Feature Selection: Bag of words, Tf-idf
Feature Extraction: Word Embedding and Dimensional Reduction
Modeling for Neural Network
"""

import collections
import struct
import sys

from os import listdir
from os.path import isfile, join

import tensorflow as tf
from tensorflow.core.example import example_pb2

from numpy.random import shuffle

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from autocorrect import spell
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

np.random.seed(20190220)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'text_to_binary',
                           'Either text_to_vocabulary or text_to_binary.'
                           'Specify FLAGS.in_directories accordingly.')
tf.app.flags.DEFINE_string('in_directories', '', 'path to directory')
tf.app.flags.DEFINE_string('out_files', '', 'comma separated paths to files')
tf.app.flags.DEFINE_string('split', '', 'comma separated fractions of data')

def csv_reader():
    return Text

def text2idx(texts):

    words = word_tokenize(sent_tokenize(texts))

    words = [spell(word) for word in words]

    stop_words = stopwords.words('english') + list(punctuation)
    words = [word.lower() for word in words]
    words = [word for word in words if w not in stop_words]

    words = words.str.replace(r"\d+", "")
    words = words.str.replace('[^\w\s]','')
    words = words.str.replace(r"[︰-＠]", "")

    stemmer = SnowballStemmer("english")
    words = [stemmer.stem(word) for word in words]

    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    



def _text_to_binary(input_directories, output_filenames, split_fractions):
  filenames = _get_filenames(input_directories)

  random_shuffle(filenames)

  start_from_index = 0
  for index, output_filename in enumerate(output_filenames):
    sample_count = int(len(filenames) * split_fractions[index])
    print(output_filename + ': ' + str(sample_count))

    end_index = min(start_from_index + sample_count, len(filenames))
    _convert_files_to_binary(filenames[start_from_index:end_index], output_filename)

    start_from_index = end_index
