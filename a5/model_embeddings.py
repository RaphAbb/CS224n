#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>'] # notice that in assignment 4 vocab is of type (Vocab), not (VocabEntry) as assignment 5.
        # self.embeddings = nn.Embedding(len(vocab.src), word_embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1h
        self.e_char = 50
        self.word_embed_size = word_embed_size
        
        self.embeddings = nn.Embedding(len(vocab.char2id), self.e_char, padding_idx=vocab.word2id['<pad>'])
        self.cnn = CNN(self.e_char, word_embed_size)
        self.highway = Highway(word_embed_size)
        self.dropout = nn.Dropout(p=0.3)
        
        
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        x_emb = self.embeddings(input) #(sentence_length, B, m_word, e_char)
        sentence_length, B, m_word, e_char = x_emb.size()
        flatten_x_emb = x_emb.view(sentence_length*B, m_word, e_char).permute(0,2,1) #(sentence_length*B, e_char, m_word)
        x_conv_out = self.cnn(flatten_x_emb) #(sentence_length*B, e_word)
        x_highway = self.highway(x_conv_out) #(sentence_length*B, e_word)
        x_word_emb = self.dropout(x_highway).view(sentence_length, B, self.word_embed_size)  #(sentence_length, B, e_word)
        
        return x_word_emb
        
        ### YOUR CODE HERE for part 1h




        ### END YOUR CODE

