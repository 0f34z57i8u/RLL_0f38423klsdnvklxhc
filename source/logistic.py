
import pandas as pd
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import make_scorer,accuracy_score,recall_score,roc_auc_score,f1_score,precision_score
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import os, logging


#Reload tensorflow model, return model weights and biases
def load_neuronet(path, model):
    tf.reset_default_graph()
    restore_graph = tf.Graph()
    w, b ={}, {}
    with tf.Session(graph=restore_graph) as restore_sess:
        restore_saver = tf.train.import_meta_graph(os.path.join(path, model))
        restore_saver.restore(restore_sess, tf.train.latest_checkpoint(path))
        g = tf.get_default_graph()
        fc_l1_query_w = g.get_tensor_by_name('fc_l1_query/weights:0').eval()
        fc_l1_query_b = g.get_tensor_by_name('fc_l1_query/biases:0').eval()
        fc_l1_doc_w = g.get_tensor_by_name('fc_l1_doc/weights:0').eval()
        fc_l1_doc_b = g.get_tensor_by_name('fc_l1_doc/biases:0').eval()

        fc_l2_query_w = g.get_tensor_by_name('fc_l2_query/weights:0').eval()
        fc_l2_query_b = g.get_tensor_by_name('fc_l2_query/biases:0').eval()
        fc_l2_doc_w = g.get_tensor_by_name('fc_l2_doc/weights:0').eval()
        fc_l2_doc_b = g.get_tensor_by_name('fc_l2_doc/biases:0').eval()
        w['fc_l1_query_w'] = fc_l1_query_w
        w['fc_l1_doc_w'] = fc_l1_doc_w
        w['fc_l2_query_w'] = fc_l2_query_w
        w['fc_l2_doc_w'] = fc_l2_doc_w

        b['fc_l1_query_b'] = fc_l1_query_b
        b['fc_l1_doc_b'] = fc_l1_doc_b
        b['fc_l2_query_b'] = fc_l2_query_b
        b['fc_l2_doc_b'] = fc_l2_doc_b
        return w, b
    
def getSigmoid(x):
    return 1.0/(1+np.exp(-x))


def forward_prop(w, b, x, which):
    if(which=='query'):
        l1_out = getSigmoid(np.dot(x, w['fc_l1_query_w'])+b['fc_l1_query_b'])
        return getSigmoid(np.dot(l1_out, w['fc_l2_query_w'])+b['fc_l2_query_b'])
    elif(which =='doc'):
        l1_out = getSigmoid(np.dot(x, w['fc_l1_doc_w'])+b['fc_l1_doc_b'])
        return getSigmoid(np.dot(l1_out, w['fc_l2_doc_w'])+b['fc_l2_doc_b'])
    
# Read and process raw data, feed to neural network
# we will use embedding from the second layer
def input_data(which, train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop(columns = ['source_name','votes', 'mv_fluency'])
    y_train = train['mv_fluency']
    X_test = test.drop(columns = ['mv_fluency', 'source_name', 'votes'])
    y_test = test['mv_fluency']
    if(which=='query'):
        X_train = forward_prop(w, b, X_train, 'query')
        X_test = forward_prop(w, b, X_test, 'query')
    elif(which=='doc'):
        X_train = forward_prop(w, b, X_train, 'doc')
        X_test = forward_prop(w, b, X_test, 'doc')
    elif(which=='both'):
        X_train = np.concatenate((forward_prop(w, b, X_train, 'query'), forward_prop(w, b, X_train, 'doc')), axis=1)
        X_test = np.concatenate((forward_prop(w, b, X_test, 'query'), forward_prop(w, b, X_test, 'doc')), axis=1)
    return X_train, y_train, X_test, y_test




train_path = '../data/raw_data/train.csv'
test_path = '../test.csv'
path = '../model/weighted/'
log_path = '../summary.log'

# result list is where you put results from different settings together
result = {}
result['accuracy'] = []
result['precision'] = []
result['recall'] = []
result['f1'] = []
result['auc'] = []
result['rll_model'] = []
result['network'] = []
result['lr_model'] = []
result['best_lr_5cv_score'] = []
result['lr_model'] = []
result['rll_model_path'] = []

# Iterate all trained RLL models, get embeddings from the model, and feed embeddings to a logistic regression
# classifier to do binary classifications on test data
for root, dirs, models in os.walk(path):
    for m in models:
        if(m[-5:]=='.meta'):
            print(os.path.join(root, m))            
            w, b = load_neuronet(root, m)
            for which in ['query', 'doc', 'both']:
                X_train, y_train, X_test, y_test = input_data(which, train_path, test_path)
                logging.basicConfig(
                    filename=log_path,
                    level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:%(message)s"
                    )

                parameters = {'penalty':['l1', 'l2'],
                                    'C':[1e-10, 1e-8, 1e-6, 1e-3, 1e-2, 1e-1, 1, 10, 1e2],
                                     'max_iter': [1e2, 1e3, 5000]
                                     }
                model =  LogisticRegression()
                clf = GridSearchCV(model, parameters, cv=4, verbose=3, scoring='accuracy')
                clf.fit(X_train, y_train)

                print('model {} {}'.format(m, which))
                logging.debug('model {} {}'.format(m, which))
                print('best 5cv score accuracy {}'.format(clf.best_score_))
                logging.debug('best 5cv score accuracy {}'.format(clf.best_score_))
                print(clf.best_params_)
                logging.debug(clf.best_params_)

                best_model = clf.best_estimator_
                y_hat = best_model.predict(X_test)
                y_hat_proba = best_model.predict_proba(X_test)

                test_conf_mat = confusion_matrix(y_test, y_hat)
                precision = precision_score(y_test, y_hat)
                recall = recall_score(y_test, y_hat)
                f1 = f1_score(y_test, y_hat)
                auc = roc_auc_score(y_test, y_hat_proba[:,1])
                accuracy = accuracy_score(y_test, y_hat)

                print('accuracy', accuracy)
                logging.debug('accuracy {}'.format(accuracy))
                logging.debug("*"*80)


                result['accuracy'].append(accuracy)
                result['precision'].append(precision)
                result['recall'].append(recall)
                result['f1'].append(f1)
                result['auc'].append(auc)
                result['rll_model'].append(m)
                result['network'].append(which)
                result['lr_model'].append(str(clf.best_params_))
                result['best_lr_5cv_score'].append(clf.best_score_)
                result['rll_model_path'].append(root)


                pd.DataFrame(result).to_excel('../summary_result_RLL.xlsx')

