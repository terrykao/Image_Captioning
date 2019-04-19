import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)


    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Create and init layers for the LSTM model.
        '''
        super(DecoderRNN, self).__init__()

        # Save passed parameters instance variables
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Determine if GPU is available, needed by init_lstm_hidden_states
        # which allocates tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding layer that converts input caption words to a vector
        # of specified embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Create the LSTM layer
        # Input: embedded vector of embed_size
        # Output: hidden states of size hidden_size
        self.lstm = nn.LSTM(input_size=self.embed_size, 
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        
        # Create the fully connected layer that maps LSTM hidden state output
        # to output vector of size vocab_size, which gives probability of most
        # likely next word in the known vocabulary
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)


    def init_lstm_hidden_states(self, batch_size):
        ''' Helper to init the LSTM hidden states to zeroes. Axes 
        have following semantics: (num_layers, batch_size, hidden_size)
        '''
        num_layers = self.num_layers
        hidden_size = self.hidden_size
        device = self.device
        
        return (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                torch.zeros(num_layers, batch_size, hidden_size).to(device))
    
    def forward(self, features, captions):
        ''' Feedforward implementation of model, using layers defined in init.
        '''
        
        # Remove <end> element
        captions = captions[:, :-1]
        
        # Get the batch size from shape of features: (batch_size, embed_size)
        #batch_size = features.size(0)
        batch_size = features.shape[0]


        # Create embedded vectors of words from input captions with shape:
        # (batch_size, len(captions)-1, embed_size)
        caption_embeds = self.embedding(captions)
        
        # Combine CNN image features and caption embeddings as input for LSTM
        # with re-shape: (batch_size, len(captions), hidden_size)
        lstm_in = torch.cat((features.unsqueeze(1), caption_embeds), dim=1)
        
        # Pass LSTM layer over the combined features and caption embeddings
        # Output shape: (batch_size, len(captions), hidden_size)
        lstm_out, self.hidden = self.lstm(lstm_in)
        
        # Pass LSTM output through the fully connected layer
        # Output shape: (batch_size, len(captions), vocab_size)
        fc_out = self.fc(lstm_out)

        #return outputs
        return fc_out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # Init captions to empty list to store each next predicted word idx
        captions = []
        
        # Zero out initial LSTM hidden states
        hidden_states = self.init_lstm_hidden_states(1)
        
        for i in range(max_len):
            # Run LSTM layer over the input image features
            lstm_out, hidden_states = self.lstm(inputs, hidden_states)
            
            # Run LSTM output through the fully connected layer
            fc_out = self.fc(lstm_out)
            
            # Find the predicted word index with highest probablity score
            predicted_word = fc_out.argmax(dim=2)
            
            scores, predicted_word_idx = torch.max(fc_out.squeeze(1), dim=1)

            # Append next predicted word index to captions list of words
            captions.append(predicted_word_idx.item())
            
            # If <end> predicted, then break out as we're done
            if predicted_word_idx.item() == 1:
                break
            
            # Prepare for next word prediction
            inputs = self.embedding(predicted_word_idx.unsqueeze(0))

        return captions