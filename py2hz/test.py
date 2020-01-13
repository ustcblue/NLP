import os
import sys
import numpy as np
from lstm_theano import LSTMTheano
import time
from datetime import datetime
import random
import utils
from utils import *


[X_train, y_train] = load_data(["../pinyin_sample_0"])

print utils.pinyin_dict

#for py in utils.pinyin_to_character:
#    print "%s\t%d" % (py, len(utils.pinyin_to_character[py]) )
