import os
import sys
import numpy as np
import utils
import heapq

import commands

from lstm_theano import LSTMTheano
#[X_train, y_train] = utils.load_data(["../pinyin_sample_test"])
[X_train, y_train, mask] = utils.load_data(["../pinyin_sample_0.new"])
zhuanming_dict = utils.load_dict("zhuanming_dict")

ins_count = len(X_train)
test_ins_count = int(ins_count * 0.01)
train_ins_count = ins_count - test_ins_count
print "ins_count: %d, test_ins_count: %d, train_ins_count: %d" % (ins_count,test_ins_count,train_ins_count)

#model = utils.load_model_parameters_theano("model/rnn-theano-80-2016-10-18-15-05-56.npz")

def get_candidate( prob , topN = 10):
    cans = {}

    for i, p in enumerate(prob):
        cans[i] = p
    
    max_can_scores = heapq.nlargest(topN,cans.values())

    max_cans = []

    for value in max_can_scores:
        if value == 0:
            break
        for idx in cans:
            if cans[idx] == value:
                max_cans.append(idx)

    return max_cans

loaded_model = ""

while True:
    line = sys.stdin.readline().strip()
    '''
    if line in zhuanming_dict:
        hanzi = ""
        ret = zhuanming_dict[line][0]
        for i in range(0,len(ret)):
            hanzi += utils.index_to_character[ ret[i] ]
            
        print hanzi
        continue
    '''
    pinyin_segs = line.split(" ")
    if len(pinyin_segs) > 0:
        input = [ ]
        oov = False
        for py in pinyin_segs:
            if py in utils.pinyin_to_index:
                input.append( utils.pinyin_to_index[py] )
            else:
                oov = True
                break

        if oov:
            print "out of vocabulary"
            continue

        (status, output) = commands.getstatusoutput("ls -l model/*.npz")
        
        latest_model = output.split(" ")[-1]

        if latest_model != loaded_model:
            model = utils.load_model_parameters_theano(latest_model)
            loaded_model = latest_model

        ret, probability = model.predict(input, mask)
        
        hanzi = ""
        cans = [ ]
        for i in range(0,len(ret)):
            hanzi += utils.index_to_character[ ret[i] ]
            can_index = get_candidate( probability[i] )
            can_ch = ""
            for c in can_index:
                can_ch += utils.index_to_character[c]
            
            print can_ch

        print hanzi
