
import json
import os
import shutil

import numpy as np
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.layers import Activation, dot, concatenate
from keras.models import Model

import tensorflow as tf
import numpy as np
import sys
from random import randint
import datetime
from sklearn.utils import shuffle
import pickle
import os
# Removes an annoying Tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def (filename, words, ):


def createTrainingMatrices(conversationFileName, wList, maxLen):
	conversationDictionary = np.load(conversationFileName).item()
	numExamples = len(conversationDictionary)
	xTrain = np.zeros((numExamples, maxLen), dtype='int32')
	yTrain = np.zeros((numExamples, maxLen), dtype='int32')
	for index,(key,value) in enumerate(conversationDictionary.iteritems()):
		# Will store integerized representation of strings here (initialized as padding)
		encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
		decoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
		# Getting all the individual words in the strings
		keySplit = key.split()
		valueSplit = value.split()
		keyCount = len(keySplit)
		valueCount = len(valueSplit)
		# Throw out sequences that are too long or are empty
		if (keyCount > (maxLen - 1) or valueCount > (maxLen - 1) or valueCount == 0 or keyCount == 0):
			continue
		# Integerize the encoder string
		for keyIndex, word in enumerate(keySplit):
			try:
				encoderMessage[keyIndex] = wList.index(word)
			except ValueError:
				# TODO: This isnt really the right way to handle this scenario
				encoderMessage[keyIndex] = 0
		encoderMessage[keyIndex + 1] = wList.index('<EOS>')
		# Integerize the decoder string
		for valueIndex, word in enumerate(valueSplit):
			try:
				decoderMessage[valueIndex] = wList.index(word)
			except ValueError:
				decoderMessage[valueIndex] = 0
		decoderMessage[valueIndex + 1] = wList.index('<EOS>')
		xTrain[index] = encoderMessage
		yTrain[index] = decoderMessage
	# Remove rows with all zeros
	yTrain = yTrain[~np.all(yTrain == 0, axis=1)]
	xTrain = xTrain[~np.all(xTrain == 0, axis=1)]
	numExamples = xTrain.shape[0]
	return numExamples, xTrain, yTrain





















"""model"""

def seq2seq_model(num_input_tokens,
                                max_input_seq_length,
                                num_target_tokens,
                                max_target_seq_length,
                                HIDDEN_UNITS,
                                input_word2idx,
                                input_idx2word,
                                target_word2idx,
                                target_idx2word):

    # Encoder
    encoder_input = Input(shape=(None,), name="encoder_input")
    encoder_input2embedding = Embedding(input_dim=num_input_tokens, output_dim=HIDDEN_UNITS, input_length=max_input_seq_length, name="encoder_input2embedding")
    embedding2lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, dropout=0.3, name="embedding2lstm")

    encoder_outputs, encoder_state_h, encoder_state_c = embedding2lstm(encoder_input2embedding(encoder_input))
    encoder_states = [encoder_state_h, encoder_state_c]

    # Decoder
    decoder_input = Input(shape=(None, num_target_tokens), name="decoder_input")
    decoder_input2lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, dropout=0.3, name="decoder_input2lstm")
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_input2lstm(decoder_input, initial_state=encoder_states)

    lstm2softmax = Dense(units=num_target_tokens, activation="softmax", name="lstm2softmax")
    softmax2output = lstm2softmax(decoder_outputs)

    # Encoder-Decoder Modelling
    model = Model(inputs=[encoder_input, decoder_input], outputs=[softmax2output])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    seq2seq_model = model

    # Encoder Modelling
    encoder_model = Model(encoder_input, encoder_states)

    # Decoder Modelling
    ddecoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
    decoder_outputs, state_h, state_c = decoder_input2lstm(decoder_input, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = lstm2softmax(decoder_outputs)
    decoder_model = Model([decoder_input] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    return seq2seq_model


"""Training"""

def model_trainer(batch_size):
