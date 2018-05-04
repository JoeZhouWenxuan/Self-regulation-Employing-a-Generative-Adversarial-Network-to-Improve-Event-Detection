import random
import numpy as np
import tensorflow as tf
import os
from collections import Counter

def pad_batch(batch, max_len=-1, shuffle=True):
    sentences_in = list(batch[0])
    targets = list(batch[1])
    if shuffle:
        sentences_in.sort(key=lambda s: -1 * len(s))
        targets.sort(key=lambda s: -1 * len(s))
    
    lens = np.array([len(s) for s in sentences_in], dtype=np.int32)
    
    if max_len == -1:
        max_len = max(lens)
    
    batch_size = len(sentences_in)
    
    sentences_in_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    targets_in_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    
    for i in range(batch_size):
        sent = sentences_in[i]
        target = targets[i]
        
        l = len(sent)
        sentences_in_batch[i, :l] = sent
        targets_in_batch[i, :l] = target
    return sentences_in_batch, targets_in_batch, lens


def uniform_tensor(shape, name, dtype=tf.float32):
    return tf.random_uniform(shape=shape, dtype=dtype, name=name)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for _ in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]       


