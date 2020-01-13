import numpy as np
from lstm_theano import LSTMTheano
import os
import sys

#import codecs

character_dict={}
pinyin_dict={}

data = []
label = []

start_token = "_START_"
end_token = "_END_"

index_to_pinyin = []
pinyin_to_index = {}

index_to_character = []
character_to_index = {}

pinyin_to_character = {}

mask = np.zeros((len(pinyin_dict),len(character_dict))).astype('int32')

def output_data_dict(filename):
    global mask
    global index_to_pinyin
    global index_to_character

    fp = open(filename,"w")
    fp.write("pinyin\t%d\n" % len(index_to_pinyin))
    fp.write("%s\n" % "\t".join(index_to_pinyin) )
    
    hz_out_str = "".join(index_to_character).encode("utf-8")
    fp.write("hanzi\t%d\n" % len(hz_out_str))
    fp.write("%s\n" % hz_out_str)
    
    fp.write("mask\n")
    for i in range(mask.shape[0]):
        fp.write("%s" % index_to_pinyin[i])
        out_str = ""
        count = 0
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                out_str += ("\t%d" % j)
                count += 1
        fp.write("\t%d\t%s\n" % (count, out_str));
    fp.close()
    
    return hz_out_str

def insert(value, list, dict, update_dict = True):
    if update_dict:
        if value not in dict:
            dict[value] = 0
        dict[value] += 1

    list.append(value)

def process_ins(line, update_dict = True):
    global start_token
    global end_token
    global pinyin_dict

    word_segs = line.split(" ")
    ret = [ ]
    
    #insert(start_token, ret, pinyin_dict)

    for w in word_segs:
        w_can = w.split(",")
        insert(w_can[0], ret, pinyin_dict, update_dict)
    
    #insert(end_token, ret, pinyin_dict)

    return ret

def process_label(line, update_dict = True):
    global end_token
    global start_token
    global character_dict

    ret = [ ]
    
    line = line.decode("utf-8")

    #insert(start_token, ret, character_dict)

    for i in range(0,len(line)):
        insert(line[i], ret, character_dict, update_dict)
    
    #insert(end_token, ret, character_dict)

    return ret

def load_data(file_list):

    global data
    global label
    
    global index_to_pinyin
    global pinyin_to_index

    global index_to_character
    global character_to_index
    
    global character_dict
    global pinyin_dict
    
    global pinyin_to_character
    
    global mask

    for f_name in file_list:
        line_idx = 0
        for line in open(f_name,"r"):
            line = line.strip()
            if line_idx % 2 == 0:
                label.append( process_label(line) )
            else:
                data.append( process_ins(line) )

                if len(data[-1]) == len(label[-1]):

                    for i,py in enumerate(data[-1]):
                        if py not in pinyin_to_character:
                            pinyin_to_character[py] = {}

                        if label[-1][i] not in pinyin_to_character[py]:
                            pinyin_to_character[py][label[-1][i]] = 0

                        pinyin_to_character[py][label[-1][i]] += 1
            
            line_idx += 1
    
    index_to_character = [x for x in character_dict]
    character_to_index = dict([(w,i) for i,w in enumerate(index_to_character)])

    index_to_pinyin = [x for x in pinyin_dict]
    pinyin_to_index = dict([(w,i) for i,w in enumerate(index_to_pinyin)])

    #X_train = np.asarray([[pinyin_to_index[w] for w in sent] for sent in data])
    #Y_train = np.asarray([[character_to_index[w] for w in sent] for sent in label])
    X_train = []
    Y_train = []

    for pinyin_sent, character_sent in zip(data,label):
        if len(pinyin_sent) == len(character_sent):
            X_train.append([ pinyin_to_index[w] for w in pinyin_sent ] )
            Y_train.append([ character_to_index[w] for w in character_sent ] )
    
    mask = np.zeros((len(pinyin_dict),len(character_dict))).astype('int32')

    for py in pinyin_to_character:
        py_idx = pinyin_to_index[py]
        for ch in pinyin_to_character[py]:
            ch_idx = character_to_index[ch]
            mask[py_idx][ch_idx] = 1

    return [np.asarray(X_train), np.asarray(Y_train), mask]

def load_dict(file_name):
    global data
    global label
    
    global index_to_pinyin
    global pinyin_to_index

    global index_to_character
    global character_to_index
    
    global character_dict
    global pinyin_dict
    
    global pinyin_to_character

    hanzi = []
    pinyin = []

    line_idx = 0
    for line in open(file_name,"r"):
        line = line.strip()
        if line_idx % 2 == 0:
            hanzi.append( process_label(line, False) )
        else:
            pinyin.append( process_ins(line, False) )

        line_idx += 1
    
    #X_train = np.asarray([[pinyin_to_index[w] for w in sent] for sent in data])
    #Y_train = np.asarray([[character_to_index[w] for w in sent] for sent in label])
    X_train = []
    Y_train = []
    zhuanming_dict = {}

    for pinyin_sent, character_sent in zip(pinyin,hanzi):
        if len(pinyin_sent) == len(character_sent):
            key = " ".join(pinyin_sent)
            if key not in zhuanming_dict:
                zhuanming_dict[key] = []
            value = []
            for w in character_sent:
                if w in character_to_index:
                    value.append(character_to_index[w])
                else:
                    value.append("OOV")

            zhuanming_dict[key].append(value)
    
    return zhuanming_dict


def save_model_parameters_theano(outfile, model):
    E, Uf, Vf, Wf, Ub, Vb, Wb = model.E.get_value(), \
            model.Uf.get_value(), model.Vf.get_value(), model.Wf.get_value(), \
            model.Ub.get_value(), model.Vb.get_value(), model.Wb.get_value()

    np.savez(outfile, E=E, Uf=Uf, Vf=Vf, Wf=Wf, Ub=Ub, Vb=Vb, Wb=Wb)
    print "Saved model parameters to %s." % outfile
   
def load_model_parameters_theano(path):
    npzfile = np.load(path)
    
    E, Uf, Vf, Wf, Ub, Vb, Wb = npzfile["E"], \
            npzfile["Uf"], npzfile["Vf"], npzfile["Wf"], \
            npzfile["Ub"], npzfile["Vb"], npzfile["Wb"]
    
    model = LSTMTheano( pinyin_dim = E.shape[1], character_dim = Vf.shape[0], hidden_dim = E.shape[0])

    model.E.set_value(E)
    model.Uf.set_value(Uf)
    model.Vf.set_value(Vf)
    model.Wf.set_value(Wf)
    
    model.Ub.set_value(Ub)
    model.Vb.set_value(Vb)
    model.Wb.set_value(Wb)
    print "Loaded model parameters from %s. hidden_dim=%d pinyin_dim=%d character_dim=%d" % (path, E.shape[0], E.shape[1], Vf.shape[0])
    return model

def output_parameter2d_trans(array, name, out_path):

    fp = open(out_path,"a")

    row, col = array.shape
    fp.write("%s\t2\t%d\t%d\t-1\n"%(name, col, row))

    for i in range(col):
        for j in range(row):
            fp.write("%f " % (array[j][i]))
        fp.write("\n")

    fp.close()


def output_parameter2d(array, name, out_path):

    fp = open(out_path,"a")

    row, col = array.shape
    fp.write("%s\t2\t%d\t%d\t-1\n"%(name, row, col))

    for i in range(row):
        for j in range(col):
            fp.write("%f " % (array[i][j]))
        fp.write("\n")

    fp.close()

def output_parameter3d(array, name, out_path):

    fp = open(out_path,"a")

    d0 , row, col = array.shape
    fp.write("%s\t3\t%d\t%d\t%d\n"%(name, d0, row, col))

    for k in range(d0):
        for i in range(row):
            for j in range(col):
                fp.write("%f " % (array[k][i][j]))
            fp.write("\n")

    fp.close()


def model_parameter_transfer(in_path, output_path):
    npzfile = np.load(in_path)
    
    E, Uf, Vf, Wf, Ub, Vb, Wb = npzfile["E"], \
            npzfile["Uf"], npzfile["Vf"], npzfile["Wf"], \
            npzfile["Ub"], npzfile["Vb"], npzfile["Wb"]
    
    output_parameter2d_trans(E,"E",output_path)
    output_parameter3d(Uf,"Uf",output_path)
    output_parameter2d(Vf,"Vf",output_path)
    output_parameter3d(Wf,"Wf",output_path)
    output_parameter3d(Ub,"Ub",output_path)
    output_parameter2d(Vb,"Vb",output_path)
    output_parameter3d(Wb,"Wb",output_path)

