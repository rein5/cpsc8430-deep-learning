import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import re
import json
import random
import pandas as pd
import pickle
import math
import sys

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("ERROR. Usage: python model_seq2seq.py testing_data_folder [output_file.txt]")
    exit()

# read input from cli
data_folder = sys.argv[1] # "./MLDS_hw2_1_data/"

output_file = "output_file.txt"
if len(sys.argv) == 3:
    output_file = sys.argv[2]

# data folders and files
#training_labels = data_folder + '/training_label.json'
#testing_labels = data_folder + '/testing_label.json'
#training_features = data_folder + '/training_data/feat/'
#testing_features = data_folder + '/testing_data/feat/'
testing_features = data_folder + '/feat/'

# vocabulary and trained model
model_file = "s2vt_e40"
vocabulary_file = "vocabulary.pickle"

# the model will be evaluated using the following ID file
#test_id_file = data_folder + '/testing_data/id.txt'
test_id_file = data_folder + '/id.txt'

# model hyperparameters 
max_caption_words = 20
input_steps = 80
output_steps = max_caption_words
feature_size = 4096
hidden_size = 640
embedding_size = 512

# preferred pytorch device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using " + device + " device") 



"""
    Utility function to preprocess a caption. 
    
    It adds a space between words and punctuation, and it gets rid of useless characters
"""
def preprocess_caption(caption):
    caption = caption.lower().strip()

    # Ref: https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    caption = re.sub(r"([?.!,¿])", r" \1 ", caption)
    caption = re.sub(r'[" "]+', " ", caption)

    # replace everything but (a-z, A-Z, ".", "?", "!", ",") to space characters
    caption = re.sub(r"[^a-zA-Z?.!,¿]+", " ", caption)
    caption = caption.strip()
    return caption

"""
    Class to compute a vocabulary of words form a dictionary of captions. 
    
    Words appearing less times than min_occurrences are discarded.
"""
class Vocabulary():
    def __init__(self, captions_dict, min_occurrences = 3):

        # collapse caption lists by 1 level (list of lists of captions to list of captions)
        captions = [caption for captions in list(captions_dict.values()) for caption in captions]

        # compute word counts and filter words based on min_occurrences
        word_counts = {}
        for caption in captions:
            words = caption.split()
            for word in words:
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1           
        self.vocabulary = sorted([word for word, count in word_counts.items() if count >= min_occurrences])
        
        # add custom tokens
        self.vocabulary += ['<BOS>', '<EOS>', '<UNK>', '<PAD>']

        # build word-to-index and index-to-word dictionaries
        vocab_size = len(self.vocabulary)
        self.itos = {i : self.vocabulary[i] for i in range(vocab_size)}
        self.stoi = {self.vocabulary[i] : i for i in range(vocab_size)}

        
"""
    PyTorch Dataset implementation for a dataset of (features, caption) pairs, 
    built from files containing labels and features

    This class is also responsible for building a vocabulary object, from the caption labels.
"""
class VideoToCaptionDataset(Dataset): 
    
    def __init__(self, labels_file, features_file, max_caption_words, vocabulary = None): 
        self.max_caption_words = max_caption_words
        self.video_id_caption_pairs = []
        self.video_id_to_features = {}
        
        print("parsing captions...", end = ' ')
        with open(labels_file, 'r') as f:
            label_data = json.load(f)
        captions_dict = {}
        for i in range(len(label_data)):
            captions_dict[label_data[i]['id']] = [preprocess_caption(caption) for caption in label_data[i]['caption']]
        print("DONE")
        
        if vocabulary == None: 
            print("building vocabulary...", end=' ')
            # build vocabulary
            self.vocab = Vocabulary(captions_dict)
            print("DONE. Vocab size: " + str(len(self.vocab.vocabulary)))
        else: 
            print("using pre-existing vocabulary")
            self.vocab = vocabulary
        
        print("building dataset...", end=' ')
        # augment captions with tokens, create dataset of (video_id, caption) pairs
        for video_id, captions in captions_dict.items():
            # features
            self.video_id_to_features[video_id] = torch.FloatTensor(np.load(features_file + video_id + ".npy"))
            
            # captions
            for caption in captions: 
                processed_caption = ['<BOS>']
                # encode caption words
                for word in caption.split():
                    if word in self.vocab.vocabulary:
                        processed_caption.append(word)
                    else:
                        processed_caption.append('<UNK>')
                if len(processed_caption)+1 > self.max_caption_words:
                    continue
                # add EOS token
                processed_caption += ['<EOS>']    
                # pad to max caption size
                processed_caption += ['<PAD>'] * (self.max_caption_words - len(processed_caption))
                processed_caption = ' '.join(processed_caption)
                
                self.video_id_caption_pairs.append((video_id, processed_caption)) 
        print("DONE. Total examples: " + str(len(self.video_id_caption_pairs)))
    
    def __len__(self):
        return len(self.video_id_caption_pairs)
    
    def __getitem__(self, idx):
        # retrieve caption and features for given index
        video_id, caption = self.video_id_caption_pairs[idx]
        feature = self.video_id_to_features[video_id]
        
        # compute one-hot tensor for the caption 
        word_ids = torch.LongTensor([self.vocab.stoi[word] for word in caption.split(' ')])
        caption_one_hot = torch.LongTensor(self.max_caption_words, len(self.vocab.vocabulary))
        caption_one_hot.zero_()
        caption_one_hot.scatter_(1, word_ids.view(self.max_caption_words, 1), 1)
        
        return {'feature': feature, 'caption_one_hot': caption_one_hot, 'caption': caption}
        


def create_trainset():
    batch_size = 64

    print("=== training set ===")
    trainset = VideoToCaptionDataset(training_labels, training_features, max_caption_words)
    trainset_loader = DataLoader(trainset, batch_size = batch_size)
    return trainset, trainset_loader





"""
    Sequence-to-sequnce model class
    
    Encoder-decoder architecture, with each side implemented with a GRU. 
    The decoder also utilizes attention. 
"""
class Seq2SeqModel(nn.Module):
    def __init__(self, vocabulary, input_steps, output_steps, feature_size, hidden_size, embedding_size, n_layers=1, dropout_p=0.45):
        super(Seq2SeqModel, self).__init__()
        
        # init vocabulary and model hyperparameters
        self.vocab = vocabulary
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.feature_size = feature_size
        self.fc_size = embedding_size 
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size 
        vocab_size = len(self.vocab.vocabulary)
        
        # first layer, before encoder
        self.fc = nn.Linear(feature_size, self.fc_size)
        self.dropout = nn.Dropout(p = dropout_p)
        
        # GRU encoder
        self.encoder = nn.GRU(self.fc_size, hidden_size, n_layers)
        
        # attention layer for decoder
        self.attention = Attention(hidden_size)
        
        # GRU decoder (with attention)
        self.decoder = nn.GRU(hidden_size*2+embedding_size, hidden_size, n_layers)
        
        # final layer: output probability distribution over vocabulary
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        
        # embedding layer for output words
        self.embedding = nn.Embedding(vocab_size, embedding_size)
    
    # function to feedforward input sequence. Returns loss against output sequence
    def forward(self, input_seq, output_seq, teacher_forcing_ratio = -1):
        # initialize loss
        loss = 0.0
        
        # get batch size from input sequence
        batch_size = input_seq.shape[1]
        
        # compute paddings for input and output sequences
        encoder_padding = Variable(torch.zeros(self.output_steps, batch_size, self.fc_size)).to(device)
        decoder_padding = Variable(torch.zeros(self.input_steps, batch_size, self.hidden_size+self.embedding_size)).to(device) 
        
        # first layer before encoder
        input_seq = self.dropout(F.leaky_relu(self.fc(input_seq))) 
        
        # concatenate with encoder padding
        encoder_input = torch.cat((input_seq, encoder_padding), 0)
        
        # GRU encoder
        encoder_output, _ = self.encoder(encoder_input) 
        
        # concatenate with decoder padding
        first_decoder_input = torch.cat((decoder_padding, encoder_output[:self.input_steps, :, :]), 2)
        
        # GRU decoder (before attention)
        first_decoder_output, z = self.decoder(first_decoder_input) 
        
        # embed reference caption to predict against
        caption_embedded = self.embedding(output_seq)
        
        # define <BOS> embedding (initial decoder input)
        bos = [self.vocab.stoi['<BOS>']] * batch_size
        bos = Variable(torch.LongTensor([bos])).resize(batch_size, 1).to(device)
        
        # iterate over output steps, predict output and update loss
        for output_step in range(self.output_steps):
            
            # decoder input for current timestep
            if output_step == 0: # start decoder by feeding in <BOS> token
                decoder_input = self.embedding(bos)
            elif random.random() <= teacher_forcing_ratio: # use teacher forcing on a few random steps
                decoder_input = caption_embedded[:, output_step-1, :].unsqueeze(1)
            else: # feed in previous decoder output word
                decoder_input = self.embedding(decoder_output.max(1)[-1].resize(batch_size, 1))
            
            # compute attention weights and context
            attention_weights = self.attention(z, encoder_output[:self.input_steps])
            context = torch.bmm(attention_weights.transpose(1, 2),
                               encoder_output[:self.input_steps].transpose(0, 1))
            
            # define decoder input (with attention)
            second_decoder_input = torch.cat((decoder_input, encoder_output[self.input_steps+output_step].unsqueeze(1), context),2).transpose(0,1)
            
            # compute decoder output
            decoder_output, z = self.decoder(second_decoder_input, z)
            decoder_output = self.softmax(self.out(decoder_output[0]))
            
            # update loss
            loss += F.nll_loss(decoder_output, output_seq[:, output_step]) / self.output_steps
        
        # return average loss
        return loss
    
    # function to predict output sequence from input sequence
    def predict(self, input_seq, beam_width=1): 

        # list of predicted words
        output_seq = [] 
        
        # compute paddings for input and output sequences
        encoder_padding = Variable(torch.zeros(self.output_steps, 1, self.fc_size)).to(device)
        decoder_padding = Variable(torch.zeros(self.input_steps, 1, self.hidden_size+self.embedding_size)).to(device) 
        
        # first layer before encoder
        input_seq = F.leaky_relu(self.fc(input_seq))
        
        # concatenate with encoder padding
        encoder_input = torch.cat((input_seq, encoder_padding), 0)
        
        # GRU encoder
        encoder_output, _ = self.encoder(encoder_input)
        
        # concatenate with decoder padding
        first_decoder_input = torch.cat((decoder_padding, encoder_output[:self.input_steps, :, :]), 2)
        
        # GRU decoder (before attention)
        first_decoder_output, z = self.decoder(first_decoder_input)
        
        # define <BOS> embedding (initial decoder input)
        bos = [self.vocab.stoi['<BOS>']]
        bos = Variable(torch.LongTensor([bos])).resize(1, 1).to(device)
        
        if beam_width > 1: # beam search
            candidates = []
            # iterate over output steps, predict output 
            for output_step in range(self.output_steps):

                # compute initial candidates
                if output_step == 0: 
                    # start decoder by feeding in <BOS> token
                    decoder_input = self.embedding(bos)

                    # compute attention weights and context
                    attention_weights = self.attention(z, encoder_output[:self.input_steps])
                    context = torch.bmm(attention_weights.transpose(1, 2),
                                       encoder_output[:self.input_steps].transpose(0, 1))

                    # define decoder input (with attention)
                    second_decoder_input = torch.cat((decoder_input, encoder_output[self.input_steps+output_step].unsqueeze(1), context),2).transpose(0,1)

                    # compute decoder output
                    decoder_output, z = self.decoder(second_decoder_input, z)
                    decoder_output = self.softmax(self.out(decoder_output[0]))
                    prob = math.e ** decoder_output

                    # select top k candidates
                    top_k_candidates, top_k_ids = prob.topk(beam_width)
                    top_k_scores = top_k_candidates.data[0].cpu().numpy().tolist()
                    candidates = top_k_ids.data[0].cpu().numpy().reshape(beam_width, 1).tolist()
                    zs = [z] * beam_width

                # compute new candidates from old ones
                else: 
                    new_candidates = []
                    # iterate over old candidates
                    for i, candidate in enumerate(candidates):
                        # feed in previous decoder output word
                        decoder_input = Variable(torch.LongTensor([candidate[-1]])).to(device).resize(1,1)
                        decoder_input = self.embedding(decoder_input)

                        # compute attention weights and context
                        attention_weights = self.attention(z, encoder_output[:self.input_steps])
                        context = torch.bmm(attention_weights.transpose(1,2),
                                            encoder_output[:self.input_steps].transpose(0,1))

                        # define decoder input (with attention)
                        second_decoder_input = torch.cat((decoder_input, encoder_output[self.input_steps+output_step].unsqueeze(1), context),2).transpose(0,1)

                        # compute decoder output
                        decoder_output, zs[i] = self.decoder(second_decoder_input, zs[i])
                        decoder_output = self.softmax(self.out(decoder_output[0]))
                        prob = math.e ** decoder_output

                        # compute score for new candidates (from current old candidate i)
                        top_k_candidates, top_k_ids = prob.topk(beam_width)
                        for k in range(beam_width):
                            score = top_k_scores[i] * top_k_candidates.data[0, k]
                            new_candidate = candidates[i] + [top_k_ids[0, k].item()] 
                            new_candidates.append([score, new_candidate, zs[i]])

                    # select top k candidates
                    new_candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
                    top_k_scores = [candidate[0] for candidate in new_candidates]
                    candidates = [candidate[1] for candidate in new_candidates]
                    zs = [candidates[2] for candidates in new_candidates]

            # convert best candidate sequence of words (output_seq)
            token_indices = [self.vocab.stoi[t] for t in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']]
            output_seq = [self.vocab.itos[int(word_index)] for word_index in candidates[0] if int(word_index) not in token_indices]
            return output_seq
        
        else: # greedy search
            pred_seq = []
            for output_step in range(self.output_steps):
                # compute initial candidates
                if output_step == 0: 
                    # start decoder by feeding in <BOS> token
                    decoder_input = self.embedding(bos)
                else: # feed in previous decoder output word
                    decoder_input = self.embedding(decoder_output.max(1)[-1].resize(1, 1))

                # compute attention weights and context
                attention_weights = self.attention(z, encoder_output[:self.input_steps])
                context = torch.bmm(attention_weights.transpose(1, 2),
                                   encoder_output[:self.input_steps].transpose(0, 1))

                # define decoder input (with attention)
                second_decoder_input = torch.cat((decoder_input, encoder_output[self.input_steps+output_step].unsqueeze(1), context),2).transpose(0,1)

                # compute decoder output
                decoder_output, z = self.decoder(second_decoder_input, z)
                decoder_output = self.softmax(self.out(decoder_output[0]))

                # append current word to output sequence 
                token_indices = [self.vocab.stoi[t] for t in ['<EOS>', '<PAD>', '<UNK>']]
                output_id = decoder_output.max(1)[-1].item()
                if output_id in token_indices:
                    break
                elif output_id != self.vocab.stoi['<BOS>']:
                    pred_seq.append(self.vocab.itos[int(output_id)])
            
            return pred_seq

"""
    Attention module class
"""
class Attention(nn.Module):
    def __init__(self, hidden_size, dropout = 0.45):
        super(Attention, self).__init__()
        self.Attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(p = dropout)
        self.hidden_size = hidden_size 
        
    def forward(self, hidden, encoder_outputs):
        attention_output = torch.bmm(encoder_outputs.transpose(0,1), hidden.transpose(0, 1).transpose(1,2))
        attention_output = F.tanh(attention_output)
        attention_weights = F.softmax(attention_output, dim = 1)
        return attention_weights
    






def train(trainset_loader, model, epochs = 40, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    
    # train and store the training loss for all epochs  
    training_loss = []
    for epoch in range(epochs):
        # set model to train mode
        model.train()
        
        # init loss for current epoch
        training_epoch_loss = 0.0

        # iterate over batches
        for batch, batch_data in enumerate(trainset_loader): 
            """
            size of batch_data items: {'feature': torch.Size([64, 80, 4096]), 
                                       'caption': list of 64 captions, each of which has length max_caption_words
                                       'caption_one_hot': torch.Size([64, 20, 2422])}
            """
            X_train = batch_data['feature'].transpose(0, 1).to(device)
            Y_train = batch_data['caption_one_hot'].to(device)
            
            # reset gradients
            optimizer.zero_grad()

            # compute loss and take optimizer step
            loss = model(X_train, Y_train.max(2)[-1], teacher_forcing_ratio = 0.05)
            loss.backward()
            optimizer.step()
            
            # update epoch loss
            training_epoch_loss += loss.item() / len(trainset_loader)
        
        # append loss for current epoch
        training_loss.append(training_epoch_loss)
        
        # save model every 10 epochs
        if (epoch % 10 == 0) or (epoch == (epochs-1)):
            torch.save(model.state_dict(), "s2vt_e"+str(epoch))
        
        # print training info
        if (epoch % 1 == 0) or (epoch == (epochs-1)):
            print("Epoch " + str(epoch) + ", train loss: " + str(training_epoch_loss) )
            
    # return list of epoch training losses
    return training_loss





# function to create the training set and train a seq2seq model
def train_model():

    trainset, trainset_loader = create_trainset()
    pickle.dump(trainset.vocab, open('vocabulary.pickle', 'wb'))

    s2vt = Seq2SeqModel(trainset.vocab, input_steps, output_steps, feature_size, hidden_size, embedding_size).to(device)

    print("Training s2vt model...")
    training_loss = train(trainset_loader, s2vt)

    return training_loss






def postprocess_caption(caption):
    caption = caption.capitalize().replace(' .', "")
    return caption 


# function to run model inference on test data
def test(model_file, vocabulary_file, testing_features, test_id_file, output_file):
    vocabulary = pickle.load(open(vocabulary_file, "rb"))

    s2vt = Seq2SeqModel(vocabulary, input_steps, output_steps, feature_size, hidden_size, embedding_size).to(device)
    s2vt.load_state_dict(torch.load(model_file))
    s2vt.to(device)

    # load input data
    input_data = {}
    test_label = pd.read_fwf(test_id_file, header=None)
    for _, row in test_label.iterrows():
        feature_file = f"{testing_features}{row[0]}.npy"
        input_data[row[0]] = torch.FloatTensor(np.load(feature_file))

    # run model on above data
    s2vt.eval()
    predictions = []
    indices = []
    for _, row in test_label.iterrows():
        input_seq = Variable(input_data[row[0]].view(-1, 1, feature_size)).to(device)
        prediction = s2vt.predict(input_seq, beam_width = 2)
        prediction = postprocess_caption(" ".join(prediction))
        predictions.append(prediction)
        indices.append(row[0])

    # save results to file
    with open(output_file, 'w') as result_file:
        for i, _ in test_label.iterrows():
            result_file.write(indices[i] + "," + predictions[i] + "\n")



###### TRAIN MODEL #####

#train_model()

########################


##### TEST MODEL #####
print("evaluating model: " + model_file + "...", end=' ')

test(model_file, vocabulary_file, testing_features, test_id_file, output_file)

print("DONE")
###########################