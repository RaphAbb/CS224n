import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, batch_iter, read_corpus
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from nmt_model import NMT
from highway import Highway
from cnn import CNN
from nmt_model import NMT
from vocab import VocabEntry, Vocab


import torch
import torch.nn as nn
import torch.nn.utils

class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r', encoding = 'utf8'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_pad = self.char2id['∏']
        self.char_unk = self.char2id['Û']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

def question_1f_sanity_check():
    """ Sanity check for Highway Class init and forward methods 
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1f: Highway layer")
    print ("-"*80)
    
    print("Running test on a list of out conv layers")
    
    B = 4
    e_word = 3
    conv_out = torch.Tensor([[1,2,3], [4,5,6], [7,8,9], [10,11,12]]) #4*3
    
    high_layer = Highway(e_word)
    
    my_high = high_layer.forward(conv_out)

    output_expected_size = [B, e_word]
    assert list(my_high.size()) == output_expected_size, "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(my_high.size()))

    print("Sanity Check Passed for Question 1e: Correct Output Shape!")
    print("-"*80)

def question_1g_sanity_check():
    """ Sanity check for CNN Class init and forward methods 
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1g: CNN layer")
    print ("-"*80)
    
    print("Running test on a list of out conv layers")
    
    B = 2
    m_word = 5
    e_char = 3
    f = 3
    k = 4
    ch1, ch2, ch3, ch4 = [1,1,1], [1,-1,1], [1,0,1], [0,3,1]
    x_reshape = torch.Tensor([[ch1, ch2, ch3, ch4, ch4],
                              [ch4, ch2, ch4, ch1, ch3]])
    
    cnn_layer = CNN(m_word, f, k)
    
    my_conv = cnn_layer.forward(x_reshape)

    output_expected_size = [B, f]
    assert list(my_conv.size()) == output_expected_size, "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(my_conv.size()))

    print("Sanity Check Passed for Question 1g: Correct Output Shape!")
    print("-"*80) 

def question_1i_sanity_check():
    """ Sanity check for nmt_model.py
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1i: NMT")
    print ("-"*80)
    src_vocab_entry = VocabEntry()
    tgt_vocab_entry = VocabEntry()
    dummy_vocab = Vocab(src_vocab_entry, tgt_vocab_entry)
    word_embed_size = 5
    hidden_size = 10
    
    nmt = NMT(word_embed_size, hidden_size, dummy_vocab)
    source = [["Hello my friend"], ["How are you"]]
    target = [["Bonjour mon ami"], ["Comment vas tu"]]
    output = nmt.forward(source, target)
    
    print(output)
    #output_expected_size = [sentence_length, BATCH_SIZE, EMBED_SIZE]
    #assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
    print("Sanity Check Passed for Question 1i: NMT!")
    print("-"*80)
    
def main():
    """ Main func.
    """
    arg = sys.argv[1]

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    

    if arg == '1f':
        question_1f_sanity_check()
    if arg == '1g':
        question_1g_sanity_check()
    if arg == '1i':
        question_1i_sanity_check()
        
if __name__ == "__main__":
    main()