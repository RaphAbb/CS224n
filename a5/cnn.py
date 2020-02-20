#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g
    def __init__(self, e_char, f, k = 5):
        '''
        Init the Convolution Neural Network
        @param e_char (int): character embedding size
        @param f (int): numbers of filter/output channels = word_embedding_size
        '''
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=e_char, out_channels=f, kernel_size=k, stride=1, padding=1, bias=True)
        #self.max_pool = nn.MaxPool1d()


    def forward(self, x_reshape):
        '''
        Compute the Convolutional Steps from reshaped character embeddings to Conv_Out
        @param x_reshape (Tensor):  (B, e_char, m_word) size tensor of reshaped char embeddings
                                    B: batch size, e_char: char embedding size , m_word: max word size
                                   
        @return x_conv_out (Tensor):    (B, f) size tensor pf convolutionned embeddings
                                        B: batch size, e_word: word emb. size, k: conv. kernal size
                                        Here we take f = e_word
        '''
        #x_reshape is transformed by a convolution into x_conv of dimension (B, f, m_word-k+1)
        #f: number of fiters = e_word: word emb. size
        #x_maxpool shrinks along the filters (taking the max of the filter column) -> (B, f)
        x_conv = self.conv(x_reshape)
        return F.max_pool1d(F.relu(x_conv), x_conv.shape[-1]).squeeze(-1)
    

    ### END YOUR CODE

