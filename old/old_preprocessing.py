"""
tf.app.flags.FLAGSを使い、TensorFlowのPythonファイルを実行する際にパラメタを付与する
下記のようにすると、パラメタ付与が可能になり、デフォルト値やヘルプ画面の説明文を登録できる

tf.app.flags.DEFINE_string('変数名', 'デフォルト値', """説明文""")

tf.app.flags.DEFINE_stringの他に、tf.app.flags.DEFINE_boolean, tf.app.flags.DEFINE_integerがある


===================================================

コードサンプル (test.py)

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('data_num', 100, """データ数""")
tf.app.flags.DEFINE_string('img_path', './img', """画像ファイルパス""")

def main(argv):
    print(FLAGS.data_num, FLAGS.img_path)

if __name__ == '__main__':
    tf.app.run()
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
from tensorflow.python.platform import gfile

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 12500, "input vocabulary size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 12500, "output vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./datas", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./datas", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
