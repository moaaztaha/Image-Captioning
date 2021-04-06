import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import random


class DecoderRNN(nn.Module):
    def __init__(self, emb_dim, hid_dim, vocab_sz, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.vocab_sz = vocab_sz

        self.embedding = nn.Embedding(vocab_sz, emb_dim)

        # LSTM: inputs-> [embeddings + context], output -> hidden, output
        self.rnn = nn.LSTM(emb_dim+hid_dim, hid_dim)

        # FC: inputs:-> [embeddings + context + output]
        self.fc_out = nn.Linear(emb_dim + hid_dim*2, vocab_sz)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

#         print(f'Context shape {context.shape}')
#         print(f'embeded shape {embedded.shape}')

        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con = [1, batch size, emb_dim + hid_dim]

#         print(f"Hidden shape {hidden.shape}")
#         print(f"embedded shape {embedded.shape}")
#         print(f"context shape {context.shape}")
#         print(f"emb_con shape {emb_con.shape}")
#         print('_'*22)

        output, hidden = self.rnn(emb_con, hidden)
        # output = [1, batch size, hid dim] -> seq len and n directions = 1
        # hidden = [1, batch size, hid dim] -> n layers and n directions =1

        output = torch.cat(
            (embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        # output = [batch size, emb_dim + hid_dim*2]

        predictions = self.fc_out(output)
        # predictions = [batch size, vocab size]

        return predictions, hidden

###
# No teacher forcing for now
###


class Img2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, tf_ratio=0.5):
        super().__init__()

        self.tf_ratio = tf_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert self.encoder.hid_dim == self.decoder.hid_dim,\
            'The hid dim of the context vector must equal the hid dim of the decoder !!'

    def forward(self, img, trg):
        # src = [batch size, 3, 224, 224]
        # trg = [trg len, batch size]

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_sz

        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        context = self.encoder(img)
        context = context.unsqueeze(0)
        hidden = context

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, context)

            outputs[t] = output

            top1 = output.argmax(1)

            teacher_force = random.random() < self.tf_ratio

            input = trg[t] if teacher_force else top1

        return outputs
