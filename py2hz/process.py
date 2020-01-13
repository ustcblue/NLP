import os
import sys
import numpy as np
from lstm_theano import LSTMTheano
import time
from datetime import datetime
import random
import utils
from utils import *

_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '100'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.001'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

def shuffle_train_ins(ins_num):
    dict = {}
    ins_idx_list = []

    while len(ins_idx_list) < ins_num:
        idx = random.randint(0, ins_num-1)
        if idx not in dict:
            dict[idx] = 1
            ins_idx_list.append(idx)
    
    return ins_idx_list

def train_with_sgd(model, X_train, y_train, X_test, y_test, mask, learning_rate=0.005, nepoch=1, evaluate_loss_after=1000):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    
    ins_idx = shuffle_train_ins(len(X_train))

    for i in range(len(y_train)):
        # Optionally evaluate the loss
        if (i % evaluate_loss_after == 0):
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: start calculating loss" % time
            loss = model.calculate_loss(X_test, y_test, mask)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d: %f" % (time, num_examples_seen, loss)
            # Adjust the learning rate if loss increases
            #if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
            #    learning_rate = learning_rate * 0.5  
            #    print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./model/rnn-theano-%d-%s.npz" % (model.hidden_dim, time), model)
        # For each training example...
            # One SGD step
        model.sgd_step(X_train[ins_idx[i]], y_train[ins_idx[i]], mask, learning_rate)
        num_examples_seen += 1


[X_train, y_train, mask] = load_data(["../pinyin_sample_0.ruby_pinyin.fix"])

print "training samples: %d" % (len(X_train))
print "pinyin vocabulary: %d" % (len(utils.pinyin_dict))
print "character vocabulary: %d" % (len(utils.character_dict))

#model = load_model_parameters_theano("model/rnn-theano-100-2016-10-20-14-52-37.npz")
model = LSTMTheano( pinyin_dim = len(utils.pinyin_dict), character_dim = len(utils.character_dict), hidden_dim =_HIDDEN_DIM)
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], mask, _LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
loss = model.calculate_loss([X_train[10]], [y_train[10]], mask)
print "single loss: %f" % loss
#if _MODEL_FILE != None:
#    load_model_parameters_theano(_MODEL_FILE, model)
percent = float(sys.argv[1])
ins_count = int(percent * len(X_train))
#ins_count = 1000
test_ins_count = int(ins_count * 0.01)
train_ins_count = ins_count - test_ins_count
print "ins_count: %d, test_ins_count: %d, train_ins_count: %d" % (ins_count,test_ins_count,train_ins_count)

train_with_sgd(model, X_train[0:train_ins_count], y_train[0:train_ins_count], X_train[train_ins_count:], y_train[train_ins_count:], mask, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)

