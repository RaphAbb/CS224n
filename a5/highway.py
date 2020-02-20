#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import VocabEntry

class Highway(nn.Module):
    ### YOUR CODE HERE for part 1f
    def __init__(self, emb_word_size: int):
        '''
        Init the Hughway layer
        @param emb_word_size (int): size of words embeddings
        
        @return void
        '''
        super(Highway, self).__init__()
        self.proj = nn.Linear(emb_word_size, emb_word_size, bias = True)
        #self.activated_proj = nn.ReLU()
        
        self.gate = nn.Linear(emb_word_size, emb_word_size, bias = True)
        #self.activated_gate = nn.ReLU()
        
    
        
    def forward(self, conv_out):
        '''
        Compute Highway layer from conv_out layer for a batch
        @param conv_out (Tensor): Convolution layer after Max Pooling (B, e_word)
                                    B: batch size, e_word: word embedding size
                        
        @return highway (Tensor): (B, e_word) highway tensor 
        '''
        activated_proj = F.relu(self.proj(conv_out))
        activated_gate = F.relu(self.gate(conv_out))
        
        highway = activated_gate*activated_proj + (1-activated_gate)*conv_out
        
        return highway

    ### END YOUR CODE