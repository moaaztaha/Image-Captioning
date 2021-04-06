
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


# LSTM + full teacher forcing
# encoder CNN
class EncoderCNN(nn.Module):
    def __init__(self, hid_dim, dropout, train_cnn=False):
        super().__init__()

        self.hid_dim = hid_dim
        self.train_cnn = train_cnn

        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, hid_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
        features = self.dropout(self.relu(self.inception(images)))
        return features


# decoder LSTM
class DecoderLSTM(nn.Module):
    def __init__(self, emb_dim, hid_dim, vocab_size, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim)
        self.linear = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embedding(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        rnn_out, _ = self.rnn(embeddings)
        outputs = self.linear(rnn_out)
        return outputs


# decoder GRU
class DecoderGRU(nn.Module):
    def __init__(self, emb_dim, hid_dim, vocab_size, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.linear = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embedding(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        rnn_out, _ = self.rnn(embeddings)
        outputs = self.linear(rnn_out)
        return outputs


class Img2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert self.encoder.hid_dim == self.decoder.hid_dim

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
