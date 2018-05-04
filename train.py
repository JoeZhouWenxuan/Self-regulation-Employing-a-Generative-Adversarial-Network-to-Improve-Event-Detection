'''
Created on 2017年10月30日

@author: zhouwenxuan
'''

import tensorflow as tf
import numpy as np
import os
from data_scripts import initialize_vocabulary, load_data, load_pretrain
from model import Model


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_, vocab = initialize_vocabulary('data/wordlist')
vocab_size = len(vocab)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('classes', 34, 'Number of classification (default: 34)')

# Model Hyperparameters
tf.app.flags.DEFINE_integer('sequence_length', 80, 'Sentence length (default: 80)')
tf.app.flags.DEFINE_integer('num_layers', 1, 'Number of hidden layers (default: 1)')
tf.app.flags.DEFINE_integer('hidden_size', 150, 'Hidden size (default: 150)')
tf.app.flags.DEFINE_float('feature_weight_dropout', 0.2, 'Feature weight dropout rate (default: 0.2)')
tf.app.flags.DEFINE_float('dropout_rate', 0, 'Dropout rate (default: 0)')
tf.app.flags.DEFINE_string("rnn_unit", "lstm", "RNN unit type (default: lstm)")
tf.app.flags.DEFINE_float('lr_decay', 0.95, 'LR decay rate (default: 0.95)')
tf.app.flags.DEFINE_float('learning_rate', 0.3, 'Learning rate (default: 0.3)')
tf.app.flags.DEFINE_float('l2_rate', 0.00, 'L2 rate (default: 0)')

# Training parameters
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs (default: 200)')
tf.app.flags.DEFINE_integer('train_max_patience', 100, 'default: 100')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Batch Size (default: 64)')
tf.app.flags.DEFINE_string("model_path", "model/best.pkl", "Path model to be saved (default: model/best.pkl)")
tf.app.flags.DEFINE_string("feature_weight_shape", "[" + str(vocab_size) + ", 300]", "Shape of feature weight table (default: [vocab_size, 300])")

FLAGS._parse_flags()
config = dict(FLAGS.__flags.items())
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("\t{} = {}".format(attr.upper(), value))
print("")

def main():

  
    train_data = load_data('data/train.txt')
    dev_data = load_data('data/dev.txt', False)
    pretrain_embedding = load_pretrain('data/wordvec')
    config['feature_init_weight'] = pretrain_embedding
    
    model = Model(config)
    train_data_sent = [item[0] for item in train_data]
    train_data_label = [item[1] for item in train_data]
    dev_data_sent = [item[0] for item in dev_data]
    dev_data_label = [item[1] for item in dev_data]
    train_data = (train_data_sent, train_data_label)
    dev_data = (dev_data_sent, dev_data_label)
    model.fit(train_data, dev_data)
    
    #predict
    # saver = tf.train.Saver()
    # saver.restore(model.sess, config['model_path'])
    # model.predict(test_data)

if __name__ == '__main__':
    main()
