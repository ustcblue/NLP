import os
import sys
import numpy as np
from lstm_theano import LSTMTheano
import time
from datetime import datetime
import random
import utils
from utils import *

import commands

[X_train, y_train, mask] = load_data(["../pinyin_sample_0"])

model_dir = "model/"

print "training samples: %d" % (len(X_train))
print "pinyin vocabulary: %d" % (len(utils.pinyin_dict))
print "character vocabulary: %d" % (len(utils.character_dict))

(status, output) = commands.getstatusoutput("ls -l " + model_dir + "*.npz")
        
latest_model = output.split(" ")[-1]

model = utils.load_model_parameters_theano(latest_model)

zhuanming_dict = utils.load_dict("zhuanming_dict")

ins_count = len(X_train)
test_ins_count = int(ins_count * 0.01)
train_ins_count = ins_count - test_ins_count
print "ins_count: %d, test_ins_count: %d, train_ins_count: %d" % (ins_count,test_ins_count,train_ins_count)

X_test = X_train[train_ins_count:]
y_test = y_train[train_ins_count:]

correct = 0
total = 0
zhuanming_hit = 0

def index_to_string(x):
    hanzi = ""
    for i in range(len(x)):
        hanzi += utils.index_to_character[x[i]]
    return hanzi

for i,x in enumerate(X_test):
    pinyin = " ".join([ utils.index_to_pinyin[py] for py in x])
    if pinyin in zhuanming_dict:
        ret = zhuanming_dict[pinyin][0]
        zhuanming_hit += 1
    else:
        ret, probability = model.predict(x,mask)

    #ret, probability = model.predict(x,mask)
    #if i > 500:
    #    break
    #print index_to_string(ret)
    #print index_to_string(y_test[i])

    for j in range(len(x)):
        if ret[j] == y_test[i][j]:
            correct += 1
        total += 1

print "total character: %d, correct character: %d, zhuanming_hit: %d, ratio: %f" % (total, correct, zhuanming_hit, correct*1.0/total)
