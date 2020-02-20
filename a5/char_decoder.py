#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size, padding_idx=self.target_vocab.char_pad)


    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        #h0, c0 = dec_hidden.squeeze(0) #Tensors: (batch_size, hidden_size)
        X = self.decoderCharEmb(input) #Tensor: (length, batch_size, e_char)
        l, b, e_char = X.size()
        enc_hiddens, (last_hidden, last_cell) = self.charDecoder(X, dec_hidden) #Tensors: (length, batch_size, hidden_size)
                                                                    # and tuple of tensors (1, batch, hidden_size)
        scores = self.char_output_projection(enc_hiddens) #Tensor: (length, batch_size, vocab_size)
        #scores_unsqueezed = scores.contiguous().view(l, b, len(self.target_vocab.char2id))
        
        return scores, (last_hidden, last_cell)
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        length, batch_size = char_sequence.size()
        input = char_sequence[:length-1,:] #remove <END> token for each elem in the batch
        target = char_sequence[1:,:] #remove <START> token
        
        scores, (last_hidden, last_cell) = self.forward(input, dec_hidden)
        
        logits = scores.contiguous().view((length-1)*batch_size, len(self.target_vocab.char2id)) # Flattened vector of ouput char distribution
        y_train = target.contiguous().view((length-1)*batch_size) #Flattened vector of input char indices
        
        loss_func = nn.CrossEntropyLoss(reduction="sum", ignore_index = self.target_vocab.char_pad)
        loss = loss_func(logits, y_train)
        
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        dec_hidden = initialStates
        _, batch_size, hidden_size = dec_hidden[0].size()
      
        output_words = [""]*batch_size
        current_chars = torch.zeros([1, batch_size], dtype = torch.long, device = device) + self.target_vocab.start_of_word # <START> token
        
        is_ended = [False]*batch_size
        
        softmax = torch.nn.Softmax(dim=2) #softmax for each batch on the hidden layer output
        for t in range(max_length):
            scores, dec_hidden = self.forward(current_chars, dec_hidden)  #Tensors: (1, batch_size, hidden_size)
                                                                                # and tuple of tensors (1, batch_size, hidden_size)
            current_chars = torch.argmax(softmax(scores), dim = 2)
            for b in range(batch_size):
                if is_ended[b]:
                    pass
                else:
                    if int(current_chars[0,b]) != self.target_vocab.end_of_word:
                        output_words[b] += self.target_vocab.id2char[int(current_chars[0,b])]
                    else:
                        is_ended[b] = True
                    
            
        #output_words = output_words[(output_words!='{') and (output_words!='}')] #remove <START> and <END> tokens
        
        return output_words
        
        ### END YOUR CODE

