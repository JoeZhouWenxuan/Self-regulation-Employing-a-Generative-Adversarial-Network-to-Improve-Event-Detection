'''
Created on 2017年10月28日

@author: zhouwenxuan
'''

import tensorflow as tf
import time
import datetime
import math
import numpy as np

from tensorflow.contrib import rnn
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score
from utils import uniform_tensor, batch_iter, pad_batch


class Model(object):
    def __init__(self, config):
        """
            config: 参数字典
                sequence_length: padding后句子的长度
                
        """
        self.sequence_length = config['sequence_length']
        self.classes = config['classes']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.feature_init_weight = config['feature_init_weight']
        
        self.feature_weight_shape = eval(config['feature_weight_shape'])
        self.feature_weight_dropout = config['feature_weight_dropout']
        self.dropout_rate = config['dropout_rate']
        self.rnn_unit = config['rnn_unit']
        self.model_path = config['model_path']
        self.l2_rate = config['l2_rate']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.train_max_patience = config['train_max_patience']
        self.max_dev_f1 = 0.0
        self.build()
        
    def build(self):
        """
            
        """
        
        self.input_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length], name='input')
        self.sequence_actual_lengths_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_actual_lengths')
        self.sequence_last_index_ph = tf.placeholder(dtype=tf.int32, name='sequence_last_index')
        self.dropout_rate_ph = tf.placeholder(tf.float32, name='dropout_rate')
        self.label_ph = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name='label')
        self.weight_dropout_ph = tf.placeholder(dtype=tf.float32,name='weight_dropout')
        
        if self.feature_init_weight is None:
            self.feature_weight = tf.Variable(
                initial_value=uniform_tensor(
                    shape=self.feature_weight_shape, 
                    name='f_W', 
                    dtype=tf.float32
                    ),
                name='feature_W',
                )
        else:
            self.feature_weight = tf.Variable(
                initial_value=self.feature_init_weight,
                name='feature_W',
                dtype=tf.float32,
                trainable=True
                )
        
        self.feature_embedding = tf.nn.dropout(
            x=tf.nn.embedding_lookup(
                self.feature_weight, 
                self.input_ph, 
                name='feature_embedding'
                ),
            keep_prob=1.0 - self.weight_dropout_ph,
            name='feature_embedding_dropout'
            )
        
        # define rnn cell               
        def cell():
            if self.rnn_unit == 'lstm':
                return rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            elif self.rnn_unit == 'gru':
                return rnn.GRUCell(self.hidden_size)
            else:
                raise ValueError('rnn unit must in (lstm, gru)')
                return
            
        
        def attn_cell():
            return rnn.DropoutWrapper(
                cell(), 
                output_keep_prob=1.0
                )
        # 
        cell_fw = rnn.MultiRNNCell(
            [attn_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
            )
        
        cell_bw = rnn.MultiRNNCell(
            [attn_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
            )
        # create gan's generator
        g_cell_fw = rnn.MultiRNNCell(
            [attn_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
            )
        
        g_cell_bw = rnn.MultiRNNCell(
            [attn_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
            )
        
        self.feature_entity_embedding = self.feature_embedding
        
        self.bi_rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, 
            cell_bw, 
            inputs=self.feature_entity_embedding, 
            sequence_length=self.sequence_actual_lengths_ph, 
            dtype=tf.float32,
            scope='bi-rnn')
        
        # outputs of the gan's generator
        self.g_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            g_cell_fw, 
            g_cell_bw, 
            inputs=self.feature_entity_embedding, 
            sequence_length=self.sequence_actual_lengths_ph, 
            dtype=tf.float32,
            scope='g_bi-rnn')
        
        
        
        self.bi_rnn_outputs_dropout = tf.nn.dropout(
            tf.concat(self.bi_rnn_outputs, axis=-1, name='bi_rnn_outputs'),
            keep_prob=1.0,
            name='bi_rnn_outputs_dropout'
            )
        
        self.g_bi_rnn_outputs_dropout = tf.nn.dropout(
            tf.concat(self.g_outputs, axis=-1, name='g_bi_rnn_outputs'),
            keep_prob=1.0,
            name='g_bi_rnn_outputs_dropout'
            )
        # outputs of sequences without paddings
        mask = tf.sequence_mask(self.sequence_actual_lengths_ph, self.sequence_length)
        self.outputs = tf.boolean_mask(self.bi_rnn_outputs_dropout, mask, name='outputs')
        g_outputs = tf.boolean_mask(self.g_bi_rnn_outputs_dropout, mask, name='g_outputs')

        self.softmax_W = tf.get_variable('softmax_W', [self.hidden_size*2, self.classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.g_softmax_W = tf.get_variable('g_softmax_W', [self.hidden_size*2, self.classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        
        self.softmax_b = tf.get_variable('softmax_b', [self.classes],  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.g_softmax_b = tf.get_variable('g_softmax_b', [self.classes],  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        
        self.softmax_W_binary = tf.get_variable('softmax_W_binary', [self.hidden_size*2, 2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.softmax_b_binary = tf.get_variable('softmax_b_binary', [2],  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        
        self.logits = tf.nn.xw_plus_b(self.outputs, self.softmax_W, self.softmax_b, name='logits')
        self.g_logits = tf.nn.xw_plus_b(g_outputs, self.g_softmax_W, self.g_softmax_b, name='g_logits')
        
        labels = tf.contrib.layers.one_hot_encoding(
                tf.boolean_mask(self.label_ph, mask),
                num_classes=self.classes
                )
        
        
        self.d_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, 
                logits=self.g_logits,
                )
            )
        
        self.g_loss = tf.reduce_mean(
            tf.multiply(labels, tf.to_float(tf.log(1- tf.nn.softmax(self.g_logits))))
            )

        # compute diff loss
        self.diff_loss = tf.norm(tf.matmul(tf.transpose(g_outputs, [1, 0]), self.outputs))     

        # all trainable variables
        all_trainable_vars = tf.trainable_variables()
        # variables related to discriminator
        vars_d = [var for var in all_trainable_vars if var.op.name == 'g_softmax_W' or var.op.name == 'g_softmax_b']
        optimizer_d = tf.train.GradientDescentOptimizer(0.1)
        self.train_op_d = optimizer_d.minimize(self.d_loss, var_list=vars_d)
        
        # variables related to generator
        vars_g = [var for var in all_trainable_vars if var.op.name in \
                  ['g_bi-rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/weights', \
                   'g_bi-rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/biases', \
                   'g_bi-rnn/bw/multi_rnn_cell/cell_0/basic_lstm_cell/weights', \
                   'g_bi-rnn/bw/multi_rnn_cell/cell_0/basic_lstm_cell/biases']
                  ]
        optimizer_g = tf.train.GradientDescentOptimizer(0.1)
        self.train_op_g = optimizer_g.minimize(self.g_loss + 0.1 * self.diff_loss, var_list=vars_g)
        

        self.logits_binary = tf.nn.xw_plus_b(self.outputs, self.softmax_W_binary, self.softmax_b_binary, name='logits_binary') 
        self.loss = self.compute_loss()
        self.l2_loss = self.l2_rate * (tf.nn.l2_loss(self.softmax_W) + tf.nn.l2_loss(self.softmax_b))
        self.total_loss = self.loss + self.l2_loss + self.diff_loss * 0.00001
        
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.total_loss)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.apply_gradients(
            grads_and_vars = grads_and_vars, 
            global_step = self.global_step
            )

        gpu_options = tf.GPUOptions(
            visible_device_list='4,5,6,7',
            allow_growth=True
            )
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options
        )
        self.sess = tf.Session(config=session_config)
        self.sess.run(tf.global_variables_initializer())
        
        
        
        
    def compute_loss(self):

        mask = tf.sequence_mask(self.sequence_actual_lengths_ph, self.sequence_length)
        labels = tf.contrib.layers.one_hot_encoding(
                tf.boolean_mask(self.label_ph, mask),
                num_classes=self.classes
                )
        labels_binary = tf.contrib.layers.one_hot_encoding(
                tf.boolean_mask(
                    tf.sign(self.label_ph),
                    mask
                    ),
                num_classes=2
                )
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, 
            logits=self.logits,
            )
        
        cross_entropy_binary = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_binary,
            logits=self.logits_binary
            )
        
        return tf.reduce_mean(cross_entropy) + tf.reduce_mean(cross_entropy_binary)
        
        
    def fit(self, training_data, dev_data):
        """
            
            参数：
                training_data： 训练集， 类型： dict    
                    key: 特征名（or label）
                    value: np.array
                
                dev_data_dict:
        
        """
        train_data_sent, train_data_label = training_data
        train_label_count = len(training_data[0])
        train_data_indices = [i for i in range(train_label_count)]
        batches = batch_iter(
            list(zip(train_data_sent, train_data_label, train_data_indices)), 
            batch_size=self.batch_size, 
            num_epochs=self.num_epochs, shuffle=True
            )
        
        self.saver = tf.train.Saver()
        train_label_count = len(training_data[0])
        batch_num_in_epoch = int(math.ceil(train_label_count / float(self.batch_size)))
        
        max_dev_f1 = self.max_dev_f1
        current_patience = 0
        
        print('pretraining...')
        train_loss = 0.0
        for _ in range(batch_num_in_epoch):
           batch_data = batches.__next__()
           batch_data_sent, batch_data_label, batch_indices = zip(*batch_data)
           batch_data = (batch_data_sent, batch_data_label)
           batch_sequences, batch_targets, batch_sequence_actual_lengths = pad_batch(batch_data, max_len=self.sequence_length)
           index = np.array(list(enumerate(batch_sequence_actual_lengths)))
           feed_dict = dict()
           feed_dict[self.input_ph] = batch_sequences 
           feed_dict[self.sequence_actual_lengths_ph] = batch_sequence_actual_lengths
           feed_dict[self.sequence_last_index_ph] = index
           feed_dict[self.weight_dropout_ph] = self.feature_weight_dropout
           feed_dict[self.dropout_rate_ph] = self.dropout_rate
           feed_dict[self.label_ph] = batch_targets     
           _, loss = self.sess.run(
               [self.train_op_d, self.d_loss], 
               feed_dict=feed_dict)
           time_str = datetime.datetime.now().isoformat()
           print("{} loss {:g}".format(time_str, loss))
        
        for step in range(self.num_epochs):
            print('\nEpoch %d / %d:' % (step+1, self.num_epochs))
            train_loss = 0.0
            for _ in range(batch_num_in_epoch):
                batch_data = batches.__next__()
                batch_data_sent, batch_data_label, batch_indices = zip(*batch_data)
                batch_data = (batch_data_sent, batch_data_label)
                batch_sequences, batch_targets, batch_sequence_actual_lengths = pad_batch(batch_data, max_len=self.sequence_length)
                index = np.array(list(enumerate(batch_sequence_actual_lengths)))
                feed_dict = dict()
                feed_dict[self.input_ph] = batch_sequences
                
                feed_dict[self.sequence_actual_lengths_ph] = batch_sequence_actual_lengths
                feed_dict[self.sequence_last_index_ph] = index
                feed_dict[self.weight_dropout_ph] = self.feature_weight_dropout
                feed_dict[self.dropout_rate_ph] = self.dropout_rate
                feed_dict[self.label_ph] = batch_targets
                
                _ = self.sess.run(self.train_op_d, feed_dict=feed_dict)
                _ = self.sess.run(self.train_op_g, feed_dict=feed_dict)
                
                _, loss, logits, bi_rnn_outputs, bi_rnn_outputs_dropout, outputs, global_step, _ = self.sess.run(
                    [self.train_op, self.loss, self.logits, self.bi_rnn_outputs,self.bi_rnn_outputs_dropout, self.outputs, self.global_step, self.feature_weight], 
                    feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()
                per_epoch_step = global_step % batch_num_in_epoch 

                print("{} step {} / {}, loss {:g}".format(time_str, per_epoch_step, batch_num_in_epoch, loss))
                train_loss += loss
                
                if global_step % 100 == 0:
                    print('dev set: ')
                    dev_f1 = self.predict(dev_data)
                    print('')
                    
                    if max_dev_f1 < dev_f1:
                        max_dev_f1 = dev_f1
                        current_patience = 0
                        self.saver.save(self.sess, self.model_path)
                        print('model has saved to %s!' % self.model_path)
                    
            train_loss /= float(batch_num_in_epoch)
            print('train_loss: ', train_loss)
            print('training set: ')
            self.predict(training_data, True)
            # print('dev set: ')
            dev_f1 = self.predict(dev_data)
            
            
            if not self.model_path:
                continue
            
            if max_dev_f1 < dev_f1:
                max_dev_f1 = dev_f1
                current_patience = 0
                
                self.saver.save(self.sess, self.model_path)
                print('model has saved to %s!' % self.model_path)
            else:
                current_patience += 1
                # 提前终止
                if self.train_max_patience and current_patience >= self.train_max_patience:
                    print()
                    return
        return
    
        
    def predict(self, data, loginfo=False):
        test_label_count = len(data[1])
        batch_num_in_epoch = int(math.ceil(test_label_count / float(self.batch_size)))
        
        sequences, targets, sequence_actual_lengths = pad_batch(data, max_len=self.sequence_length, shuffle=False)
        all_predicts = []
        binary_all_predicts = []
        predict_labels = []
        labels = []
        binary_labels = []
        total_loss = 0.0
        for i in range(batch_num_in_epoch):
            feed_dict = dict()
            batch_indices = np.arange(
                i*self.batch_size, 
                (i+1)*self.batch_size if (i+1)*self.batch_size <= test_label_count else test_label_count)
            
            feed_dict[self.input_ph] = sequences[batch_indices]
            feed_dict[self.sequence_actual_lengths_ph] = sequence_actual_lengths[batch_indices]
            
            index = np.array(list(enumerate(sequence_actual_lengths[batch_indices])))
            feed_dict[self.sequence_last_index_ph] = index
            
            feed_dict[self.weight_dropout_ph] = 0.0
            feed_dict[self.dropout_rate_ph] = 0.0
            batch_targets = targets[batch_indices]
            feed_dict[self.label_ph] = batch_targets

            
            logits, loss = self.sess.run(
                [self.logits, self.loss],
                feed_dict=feed_dict
                )
            total_loss += loss
            predicts = np.argmax(logits, axis=-1)
            
            all_predicts.extend(predicts)
            binary_all_predicts.extend(list(np.sign(predicts)))
            last_index = 0
            for j, length in enumerate(sequence_actual_lengths[batch_indices]):
                predict_labels.append(predicts[last_index:last_index + length])
                last_index += length
                labels.extend(list(batch_targets[j, :length]))
                binary_labels.extend(list(np.sign(batch_targets[j, :length])))  
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, 
            all_predicts, 
            labels=list(range(1, self.classes)), 
            average='micro'
            )
        
        binary_precision = precision_score(binary_labels, binary_all_predicts, average="binary")
        binary_recall = recall_score(binary_labels, binary_all_predicts, average="binary")
        binary_f1 = f1_score(binary_labels, binary_all_predicts, average="binary")
        if loginfo:
            print("multi-classification precision {:g}, recall {:g}, f1 {:g}".format(precision, recall, f1))
            print("binary-classification precision {:g}, recall {:g}, f1 {:g}".format(binary_precision, binary_recall, binary_f1))
        return f1
            
