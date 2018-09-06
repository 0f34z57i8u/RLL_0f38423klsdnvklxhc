import pickle
import pandas as pd
from scipy import sparse
import collections
import random
import time
import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import tensorflow.contrib.layers as layers


#read input data and write to dict
def read_input_feature(fileList, feature_path, weight_path):
    data = {}
    for f in fileList:
        if(os.path.exists(os.path.join(feature_path, f))):
            df = pd.read_csv(os.path.join(feature_path, f))
            data[f[:-4]] = np.array(df, dtype=np.float32)

        if(os.path.exists(os.path.join(weight_path, f))):
            df = pd.read_csv(os.path.join(weight_path, f))['votes']
            data[f[:-4]] = np.array(df, dtype=np.float32)
    return data


# get batch data from dict
def getBatch(data, bs, idx):
    batch_dict = []
    for k, v in data.items():
        batch_dict.append(v[bs*idx : bs*(idx+1)])
    return batch_dict


#compute cosine distance between two vectors
def cos_sim(a, b):
    normalize_a = tf.nn.l2_normalize(a,1)        
    normalize_b = tf.nn.l2_normalize(b,1) 
    cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b), axis=1)
    return cos_similarity


'''
Parameters

bs: batch size
lr_rate: learning rate
l1_n: number of neurons in the first layer
l2_n: number of neurons in the second layer
max_iter: max interation number for training
reg_scale: regularization penalty
dropout_rate: ratio to drop neurons
gamma: a hyperparameter in loss function, should be fine using 10.0
weighted: True or False. If true, weights(based on votes of an example) are integrated into loss function
save_path: where you save the model
model_name: you can name your model however you like
'''
def RLL(train_data, test_data, bs, lr_rate, l1_n, l2_n, max_iter, reg_scale, dropout_rate, gamma, weighted, save_path, model_name):
    tf.reset_default_graph()
    dimension = train_data['query_train'].shape[1]
    is_training=tf.placeholder_with_default(False, shape=(), name='is_training')
    queryBatch = tf.placeholder(tf.float32, shape=[None, dimension], name='queryBatch')
    posDocBatch = tf.placeholder(tf.float32, shape=[None, dimension], name='posDocBatch')
    negDocBatch0 = tf.placeholder(tf.float32, shape=[None, dimension], name='negDocBatch0')
    negDocBatch1 = tf.placeholder(tf.float32, shape=[None, dimension], name='negDocBatch1')
    negDocBatch2 = tf.placeholder(tf.float32, shape=[None, dimension], name='negDocBatch2')
    posDocWeight = tf.placeholder(tf.float32, shape=[None], name='posDocWeight')
    negDocWeight0 = tf.placeholder(tf.float32, shape=[None], name='negDocWeight0')
    negDocWeight1 = tf.placeholder(tf.float32, shape=[None], name='negDocWeight1')
    negDocWeight2 = tf.placeholder(tf.float32, shape=[None], name='negDocWeight2')

    with tf.name_scope('fc_l1_query'):
        query_l1_out = tf.contrib.layers.fully_connected(queryBatch, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                             activation_fn = tf.nn.sigmoid, scope='fc_l1_query')
        query_l1_out=tf.layers.dropout(query_l1_out, dropout_rate, training=is_training)
    with tf.name_scope('fc_l1_doc'):
        pos_doc_l1_out =  tf.contrib.layers.fully_connected(posDocBatch, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                            activation_fn = tf.nn.sigmoid, scope='fc_l1_doc')
        pos_doc_l1_out=tf.layers.dropout(pos_doc_l1_out, dropout_rate, training=is_training)

        neg_doc0_l1_out =  tf.contrib.layers.fully_connected(negDocBatch0, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                             activation_fn = tf.nn.sigmoid, scope='fc_l1_doc', reuse=True)
        neg_doc0_l1_out=tf.layers.dropout(neg_doc0_l1_out, dropout_rate, training=is_training)

        neg_doc1_l1_out =  tf.contrib.layers.fully_connected(negDocBatch1, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                              activation_fn = tf.nn.sigmoid, scope='fc_l1_doc', reuse=True)
        neg_doc1_l1_out=tf.layers.dropout(neg_doc1_l1_out, dropout_rate, training=is_training)

        neg_doc2_l1_out =  tf.contrib.layers.fully_connected(negDocBatch2, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                              activation_fn = tf.nn.sigmoid, scope='fc_l1_doc', reuse=True)

        neg_doc2_l1_out=tf.layers.dropout(neg_doc2_l1_out, dropout_rate, training=is_training)


    with tf.name_scope('fc_l2_query'):
        query_l2_out = tf.contrib.layers.fully_connected(query_l1_out, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                         activation_fn = tf.nn.sigmoid, scope='fc_l2_query')
        query_l2_out=tf.layers.dropout(query_l2_out, dropout_rate, training=is_training)
    with tf.name_scope('fc_l2_doc'):
        pos_doc_l2_out = tf.contrib.layers.fully_connected(pos_doc_l1_out, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale), 
                                                           activation_fn = tf.nn.sigmoid, scope='fc_l2_doc')
        pos_doc_l2_out=tf.layers.dropout(pos_doc_l2_out, dropout_rate, training=is_training)

        neg_doc0_l2_out = tf.contrib.layers.fully_connected(neg_doc0_l1_out, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                            activation_fn = tf.nn.sigmoid, scope='fc_l2_doc', reuse=True)
        neg_doc0_l2_out=tf.layers.dropout(neg_doc0_l2_out, dropout_rate, training=is_training)

        neg_doc1_l2_out = tf.contrib.layers.fully_connected(neg_doc1_l1_out, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                            activation_fn = tf.nn.sigmoid, scope='fc_l2_doc', reuse=True)
        neg_doc1_l2_out=tf.layers.dropout(neg_doc1_l2_out, dropout_rate, training=is_training)

        neg_doc2_l2_out = tf.contrib.layers.fully_connected(neg_doc2_l1_out, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale),\
                                                            activation_fn = tf.nn.sigmoid, scope='fc_l2_doc', reuse=True)

        neg_doc2_l2_out=tf.layers.dropout(neg_doc2_l2_out, dropout_rate, training=is_training)

    if(weighted):
        with tf.name_scope('loss'):
            reg_ws_0 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_query')
            reg_ws_1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_doc')
            reg_ws_2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_query')
            reg_ws_3 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_doc')
            reg_loss = tf.reduce_sum(reg_ws_0)+tf.reduce_sum(reg_ws_1)+tf.reduce_sum(reg_ws_2)+tf.reduce_sum(reg_ws_3)

            nominator = tf.multiply(posDocWeight, tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, pos_doc_l2_out))))
            doc0_similarity = tf.multiply(negDocWeight0, tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, neg_doc0_l2_out))))
            doc1_similarity = tf.multiply(negDocWeight1, tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, neg_doc1_l2_out))))
            doc2_similarity = tf.multiply(negDocWeight2, tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, neg_doc2_l2_out))))
            prob = tf.add(nominator,tf.constant(1e-10))/tf.add(doc0_similarity+ doc1_similarity+doc2_similarity+nominator,tf.constant(1e-10))
            log_prob = tf.log(prob)
            loss_batch = -tf.reduce_sum(log_prob) + reg_loss
    else:
        with tf.name_scope('loss'):
            reg_ws_0 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_query')
            reg_ws_1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_doc')
            reg_ws_2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_query')
            reg_ws_3 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_doc')
            reg_loss = tf.reduce_sum(reg_ws_0)+tf.reduce_sum(reg_ws_1)+tf.reduce_sum(reg_ws_2)+tf.reduce_sum(reg_ws_3)

            nominator = tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, pos_doc_l2_out)))
            doc0_similarity = tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, neg_doc0_l2_out)))
            doc1_similarity = tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, neg_doc1_l2_out)))
            doc2_similarity = tf.exp(tf.multiply(gamma, cos_sim(query_l2_out, neg_doc2_l2_out)))
            prob = tf.add(nominator, tf.constant(1e-10))/tf.add(doc0_similarity+ doc1_similarity+doc2_similarity+nominator,tf.constant(1e-10))
            log_prob = tf.log(prob)
            loss_batch = -tf.reduce_sum(log_prob) + reg_loss
            
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdadeltaOptimizer(lr_rate).minimize(loss_batch)
        


    train_size = train_data['query_train'].shape[0]
    test_size = test_data['query_test'].shape[0]
    print('training data size ', train_size)
    print('testing data size', test_size)
    
    best_test_loss = 2147483647
    num_batch = train_size//bs
    earlyStopCount = 0
    saver = tf.train.Saver(max_to_keep=2)
    
    start = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(0, max_iter):
            total_loss = 0
            for batch in range(num_batch):
                query_batch, pos_doc_batch, neg_doc0_batch, neg_doc1_batch, \
                        neg_doc2_batch, w_pos_doc, w_neg_doc0, w_neg_doc1, w_neg_doc2 = getBatch(train_data, bs, batch)
                feed = {    is_training: True,\
                            queryBatch: query_batch, \
                            posDocBatch : pos_doc_batch, \
                            negDocBatch0 :neg_doc0_batch, \
                            negDocBatch1 : neg_doc1_batch, \
                            negDocBatch2: neg_doc2_batch,\
                            posDocWeight: w_pos_doc.reshape(-1, ),    \
                            negDocWeight0: w_neg_doc0.reshape(-1, ),\
                            negDocWeight1: w_neg_doc1.reshape(-1, ),\
                            negDocWeight2: w_neg_doc2.reshape(-1, )
                }
                _, batch_loss = sess.run([optimizer, loss_batch], feed_dict = feed)
                total_loss += batch_loss

            print("Epoch {} train loss {}, avg train loss {}".format(epoch, total_loss, total_loss/train_size))
            logging.debug("Epoch {} train loss {}, avg train loss {}".format(epoch, total_loss, total_loss/train_size))                

            if(epoch % 10==0):
                query_ts, pos_doc_ts, neg_doc0_ts, neg_doc1_ts, neg_doc2_ts, pos_doc_weight_ts, \
                        neg_doc0_weight_ts, neg_doc1_weight_ts, neg_doc2_weight_ts = getBatch(test_data, test_size, 0)
                feed_ts = {
                            is_training: False,
                            queryBatch: query_ts, 
                            posDocBatch : pos_doc_ts, 
                            negDocBatch0 :neg_doc0_ts, 
                            negDocBatch1 : neg_doc1_ts, 
                            negDocBatch2: neg_doc2_ts,
                            posDocWeight: pos_doc_weight_ts.reshape(-1, ),    
                            negDocWeight0: neg_doc0_weight_ts.reshape(-1, ),
                            negDocWeight1: neg_doc1_weight_ts.reshape(-1, ),
                            negDocWeight2: neg_doc2_weight_ts.reshape(-1, )}

                test_loss = loss_batch.eval(feed_dict = feed_ts)

                if(test_loss < best_test_loss):
                    best_test_loss = test_loss
                    earlyStopCount = 0
                    # Only save the best test performance model
                    print('dropout{}, regularization{} learning rate{} batch size {}'.format(dropout_rate, reg_scale, lr_rate, bs))
                    saver.save(sess, os.path.join(save_path, model_name), global_step=epoch)
                    print('model saved in {}'.format(save_path))
                    logging.debug('model saved in {}'.format(save_path))
                    print('collapsed time {} min'.format(round((time.time()-start)/60, 2)))
                    logging.debug('collapsed time {} min'.format(round((time.time()-start)/60, 2)))
                else:
                    earlyStopCount += 1
                print("*"*80)
                print("Epoch {} test loss {}, avg test loss {}".format(epoch, test_loss, test_loss/test_size))
                print("*"*80)
                logging.debug("*"*80)
                logging.debug("Epoch {} test loss {}, avg test loss {}".format(epoch, test_loss, test_loss/test_size))
                logging.debug("*"*80)

            if(earlyStopCount >=5):
                print('Early stop at epoch {}, test loss {}'.format(epoch, best_test_loss))
                logging.debug('Early stop at epoch {}, test loss {}'.format(epoch, best_test_loss))
                print('collapsed time {} min'.format(round((time.time()-start)/60, 2)))
                logging.debug('collapsed time {} min'.format(round((time.time()-start)/60, 2)))
                break


# This is where you state input path and weight path
feature_path = '../data/input/'
weight_path = '../data/sample_weight/bayes_infer'

#In each path above you should have the following files ready, which will be fed to the neural network
trainFileList = ['query_train.csv', 'pos_doc_train.csv', 'neg_doc0_train.csv', 'neg_doc1_train.csv', \
            'neg_doc2_train.csv', 'weight_positive_doc_train.csv', 'weight_negative_doc0_train.csv',\
            'weight_negative_doc1_train.csv', 'weight_negative_doc2_train.csv']

testFileList = ['query_test.csv', 'pos_doc_test.csv', 'neg_doc0_test.csv', 'neg_doc1_test.csv',\
               'neg_doc2_test.csv', 'weight_positive_doc_test.csv', 'weight_negative_doc0_test.csv',\
               'weight_negative_doc1_test.csv', 'weight_negative_doc2_test.csv']


train_data = read_input_feature(trainFileList, feature_path, weight_path)
test_data = read_input_feature(testFileList, feature_path, weight_path)


# do grid search
l1_n_lst = [64, 32]
l2_n_lst = [32, 16]
dropout_rate_lst = [0.5, 0.6, 0.7]
reg_scale_lst = [0.7, 0.9, 1.0, 1.5]
lr_rate_lst = [0.05, 0.01, 0.1]
bs_lst = [512, 1024]
gamma_lst = [1.0, 10.0]
max_iter = 150

for l1_n, l2_n, dropout_rate, reg_scale, lr_rate, bs, gamma in [(l1_n, l2_n, dropout_rate, \
    reg_scale, lr_rate, bs, gamma) for l1_n in l1_n_lst for l2_n in l2_n_lst for dropout_rate in dropout_rate_lst  \
    for reg_scale in reg_scale_lst for lr_rate in lr_rate_lst for bs in bs_lst for gamma in gamma_lst]:

    print('dropout{}, regularization{} learning rate{} batch size {}'.format(dropout_rate, reg_scale, lr_rate, bs))
    model_name = '_'.join([str(l1_n), str(l2_n), str(dropout_rate), str(reg_scale), str(lr_rate), str(bs), str(gamma)])
    save_path = '../model/weighted/'
    log_path = "../log/weighted/"
    save_path = os.path.join(save_path, model_name)
    log_path = os.path.join(log_path, model_name)

    if(not os.path.exists(log_path)):
        os.makedirs(log_path)

    if(not os.path.exists(save_path)):
        os.makedirs(save_path)

    logging.basicConfig(
        filename=os.path.join(log_path,'{}.log'.format(model_name)),
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s"
        )

    RLL(train_data, test_data, bs, lr_rate, l1_n, l2_n, max_iter, reg_scale, dropout_rate, gamma, True, save_path, model_name+'.ckpt')
