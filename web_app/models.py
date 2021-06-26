
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

## Show, attend and Tell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super().__init__()

        resnet = models.resnet101(pretrained=True)

        # remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):

        # [batch size, resnet hid, image_size/32, image_size/32]
        out = self.resnet(images)
        # [batch size, resnet hid, encoded image size, encoded image size]
        out = self.adaptive_pool(out)
        # [batch size, encoded image size, encoded image size, resnet hid]
        out = out.permute(0, 2, 3, 1)

        return out

    def fine_tune(self, fine_tune=True):
        # allow or pervent the computations of gradients for the conv blocks 2 through 4 of the encoder
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()

        # transforms encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # transforms decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # calculates values to be percentages (before softmax)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # calculates weightes(probabilites)

    def forward(self, encoder_out, decoder_hidden):

        
        att1 = self.encoder_att(encoder_out) # [batch size, num pixels, attention dim]
        att2 = self.decoder_att(decoder_hidden)  # [batch size, attention dim]
        att = self.full_att(self.relu(att1+att2.unsqueeze(1))).squeeze(2)  # [batch size, num pixels]
        alpha = self.softmax(att)  # [batch size, num pixels]
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [batch size, encoder dim]

        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.deocder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)
    
    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):

        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out) # [batch size, decoder dim]
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, captions_lengths):
        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # flatten the image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) # [batch size, num pixels, encoder_dim]
        num_pixels = encoder_out.size(1)

        # sort input data by decreasing lengths
        caption_lengths, sort_ind = captions_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # embedding
        embeddings = self.embedding(encoded_captions) # [batch size, max cap length, embed dim]

        # initialize LSTM state
        h, c = self.init_hidden_state(encoder_out) # [batch size, decoder dim]

        decode_lengths = (caption_lengths - 1).tolist()
        
        
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind