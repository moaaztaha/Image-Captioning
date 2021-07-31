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


class CaptionDataset(Dataset):
    """
    Caption Dataset Class
    """

    def __init__(self, imgs_dir, captions_file, vocab, transforms=None, split='train'):
        """
        :param imgs_dir: folder where images are stored
        :param captions_file: the df file with all caption information
        :param vocab: vocabuary object
        :param transforms: image transforms pipeline
        :param split: data split
        """

        # split has to be one of {'train', 'val', 'test'}
        assert split in {'train', 'val', 'test'}

        self.imgs_dir = imgs_dir
        self.df = pd.read_json(captions_file)
        self.df = self.df[self.df['split'] == split]
        self.vocab = vocab
        self.transforms = transforms
        self.split = split

        self.dataset_size = self.df.shape[0]
        # printing some info
        print(f"Dataset split: {split}")
        print(f"Unique images: {self.df.file_name.nunique()}")
        print(f"Total size: {self.dataset_size}")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):

        # loading the image
        img_id = self.df['file_name'].values[index]
        img = Image.open(self.imgs_dir+img_id).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = transfroms.ToTensor()(img)

        # loading current caption
        cap_len = self.df['tok_len'].values[index] + 2 # <sos> and <eos>
        tokens = self.df['tokens'].values[index]
        caption = torch.LongTensor(self.vocab.numericalize(tokens, cap_len))

        if self.split is 'train':
            return img, caption, cap_len
        else:
            # for val and test return all captions for calculate the bleu scores
            captions_tokens = self.df[self.df['file_name'] == img_id].tokens.values
            captions_lens = self.df[self.df['file_name'] == img_id].tok_len.values
            all_tokens = []
            for token, cap_len in zip(captions_tokens, captions_lens):
                all_tokens.append(self.vocab.numericalize(token, cap_len)[1:]) # remove <sos>

            return img, caption, cap_len, torch.tensor(all_tokens), img_id


def caption_image(loader, vocab, encoder, decoder, beam_size):
    vocab_size = len(vocab)
    references = list()
    hypotheses = list()
    img_ids = list()

    # For each image
    for i, (image, caps, caplens, allcaps, img_id) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size), position=0, leave=True)):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[vocab.stoi['<sos>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            #             print(top_k_scores, top_k_words)
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != vocab.stoi['<eos>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        if len(complete_seqs_scores) == 0:
            continue
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {vocab.stoi['<sos>'], vocab.stoi['<eos>'], vocab.stoi['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {vocab.stoi['<sos>'], vocab.stoi['<eos>'], vocab.stoi['<pad>']}])

        img_ids.append(img_id[0])
        assert len(references) == len(hypotheses) == len(img_ids)
    # Calculate BLEU-4 scores
    #     bleu4 = corpus_bleu(references, hypotheses)
    return references, hypotheses, img_ids
    # print_scores(references, hypotheses, nltk=True)