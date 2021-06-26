import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

from dataset import build_vocab
from utils import *
from models import *

import sys



# transforms
tt = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


vocab = build_vocab('data.json')
checkpoint = load_checkpoint('models/BEST_checkpoint_flickr8k_5_cap_per_img_2_min_word_freq_resnet101_fullvocab_fix_ds_2.pth.tar', cpu=True)

device = torch.device( 'cpu')


encoder = checkpoint['encoder'].to(device)
decoder = checkpoint['decoder'].to(device)
img_path = sys.argv[1]

#def cap_image(encoder, decoder, image_path, vocab):
vocab_size = len(vocab)


img = Image.open(img_path).convert("RGB")
img = tt(img).unsqueeze(0) # transform and batch
img = img.to(device)

#encoder
encoder_out = encoder(img) # [1, enc img size, enc img size, encoder_dim]
enc_img_size = encoder_out.size(1)
encoder_dim = encoder_out.size(3)

#flatten encoder output
encoder_out = encoder_out.view(1, -1, encoder_dim) # [1, num pixels, encoder dim]
num_pixels = encoder_out.size(1)

prev_word = torch.LongTensor([vocab.stoi['<sos>']])

print(f"prev word {prev_word.item()}, {vocab.stoi['<sos>']}")

seq = list()

step = 1
h, c = decoder.init_hidden_state(encoder_out)

while True:
    embeddings = decoder.embedding(prev_word).squeeze(1)
    awe, alpha = decoder.attention(encoder_out, h)

    alpha = alpha.view(-1, enc_img_size, enc_img_size)

    gate = decoder.sigmoid(decoder.f_beta(h))
    awe = gate * awe

    h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

    scores = decoder.fc(h)
    scores = F.log_softmax(scores, dim=1)

    top = scores.argmax(1)
    
    if vocab.itos[top.item()] == '<eos>' or step > 50:
        break    
    step += 1
    seq.append(top.item())
    prev_word = top

print(" ".join([vocab.itos[idx] for idx in seq]))
show_image(image=img_path, file_name=True)