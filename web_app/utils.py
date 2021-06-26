import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
# from nltk.translate.meteor_score import meteor_score
# from torchtext.data.metrics import bleu_score
import torch.optim as optim

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os, json
from collections import Counter
from dataset import Vocabulary

from tqdm import tqdm


# transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def show_image(image=None, file_name=False, root_dir=None):
    if file_name:
        if root_dir:
            img = Image.open(root_dir+image).convert("RGB")
        img = Image.open(image).convert("RGB")
        plt.imshow(img)
        plt.show()
    else:        
        img = image.permute(1, 2, 0)
        print(img.size)
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def caption_image(image, model, vocab, max_len=100):

    model.eval()
    result_caption = []

    with torch.no_grad():
        context = model.encoder(image.to(model.device)).unsqueeze(0)
        # hidden = torch.tensor(vocab.stoi["<sos>"]).unsqueeze(0).to(model.device)
        states = None

        for _ in range(1, max_len):
            output, states = model.decoder.rnn(context, states)
            output = model.decoder.linear(output.squeeze(0))
            top1 = output.argmax(1)
            context = model.decoder.embedding(top1).unsqueeze(0)

            result_caption.append(top1.item())
            if vocab.itos[top1.item()] == '<eos>':
                break
    return [vocab.itos[idx] for idx in result_caption]


def print_examples(model, csv_name, vocab, root_dir='test_examples'):
    df = pd.read_csv(csv_name)
    imgs = df['image'].tolist()
    captions = df['description'].tolist()
    i = 1
    for img_id, cap in zip(imgs, captions):
        img = Image.open(root_dir+img_id).convert("RGB")
        plt.imshow(img)
        plt.title(f'Example {i} Correct: {cap}')
        plt.axis('off')
        img = transform(img).unsqueeze(0)
        print(f"Output: {' '.join(caption_image(img, model, vocab)[1:-1])}")
        plt.show()
        i += 1


def get_test_data(df_path):
    df = pd.read_csv(df_path)
    test_df = df[df['split'] == 'test']

    img_ids = test_df.file_name.unique()

    test_dict = {}
    for img_id in img_ids:
        list_tokens = []
        for sent in test_df[test_df['file_name'] == img_id]['caption'].values:
            list_tokens.append(Vocabulary.tokenize_en(sent))

        test_dict[img_id] = list_tokens

    return test_dict


def predict_test(test_dict, imgs_path, model, vocab, max_len=100, n_images=100):

    trgs = []
    pred_trgs = []

    i = 0

    for filename in test_dict:
        if i == n_images:
            break

        # getting the test image
        img = Image.open(imgs_path+'/'+filename).convert("RGB")
        img = transform(img).unsqueeze(0)  # making it into a batch

        # making prediction
        pred = caption_image(img, model, vocab)

        if i % (n_images//10) == 0 and i != 0:
            print("prediction:", ' '.join(x for x in pred[1:-1]))
            print("actaul 1:", ' '.join(x for x in test_dict[filename][0]))

        pred_trgs.append(pred[:-1])
        trgs.append(test_dict[filename])

        i += 1

    return pred_trgs, trgs


def print_scores(trgs, preds, vocab=None):
    print('----- Bleu-n Scores -----')
    b1 = corpus_bleu(trgs, preds, weights=[1.0/1.0])*100
    b2 = corpus_bleu(trgs, preds, weights=[1.0/2.0, 1.0/2.0])*100
    b3 = corpus_bleu(trgs, preds, weights=[1.0/3.0, 1.0/3.0, 1.0/3.0])*100
    b4 = corpus_bleu(trgs, preds)*100
    print("1:", b1)
    print("2:", b2)
    print("3:", b3)
    print("4:", b4)
    print('-'*25)
    # print("----- METEOR Score -----")
    # ids to words

    # if vocab != None:
    #     preds = [" ".join(word for word in sent) for sent in vocab.indextostring(preds)]
    #     rs = []
    #     for r in trgs:
    #         rs.append([" ".join(word for word in sent) for sent in vocab.indextostring(r)])
    #     trgs = rs
    # else:
    #     preds = [" ".join(word for word in sent) for sent in preds]
    #     rs = []
    #     for r in trgs:
    #         rs.append([" ".join(word for word in sent) for sent in r])
    #     trgs = rs
        
    
    # total_meteor = 0
    # for r, h in tqdm(zip(trgs, preds), total=len(trgs)):
    #     total_meteor += meteor_score(r, h)
    # m = total_meteor/len(rs)
    # print("m:", m)
    return b1, b2, b3, b4
    # else:
    #     print("1:", bleu_score(preds, trgs, max_n=1, weights=[1])*100)
    #     print("2:", bleu_score(preds, trgs, max_n=2, weights=[.5, .5])*100)
    #     print("3:", bleu_score(preds, trgs, max_n=3, weights=[.33, .33, .33])*100)
    #     print("4:", bleu_score(preds, trgs)*100)
    #     print('-'*25)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# show, attend, and tell
def init_embedding(embeddings):
    # fills embedding tensor with values from uniform distribution
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4, is_best):
    state = {
        'epoch': epoch,
        'epochs_since_imrovment': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'deocder_optimizer': decoder_optimizer,
    }

    file_name = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, file_name)

    if is_best:
        torch.save(state, 'BEST_' + file_name)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    # shrink learning rate by a specified factor
    print('Decaying learning rate')

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor

    print(f"The new learning rate is {optimizer.param_groups[0]['lr']}")


def accuracy(scores, targets, k):
    # compute top-k accuracy from predicted and true labels
    batch_size = targets.size(0)
    # _, ind = scores.top(k, 1, True, True)
    _, ind = torch.topk(scores, 5)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def load_checkpoint(path, cpu=False):

    if cpu:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else: 
        checkpoint = torch.load(path)
    
    print('Loaded Checkpoint!!')
    last_epoch = checkpoint['epoch']
    best_bleu4 = checkpoint['bleu-4']

    print(f"Last Epoch: {last_epoch}\nBest Bleu-4: {best_bleu4}")

    return checkpoint


def build_dataset(file_path='dataset_coco.json', df_name='coco.json'):

    with open(file_path, 'r') as f:
        data = json.load(f)

    file_names = []
    splits = []
    captions = []
    tokens = []
    tok_len = []
    word_freq = Counter()
    max_len = 100

    for img in tqdm(data['images'], position=0):
        for sent in img['sentences']:
            file_names.append(img['filename'])
            captions.append(sent['raw'])
            splits.append(img['split'])
            
            ## tokens
            if len(sent['tokens']) <= max_len:
                tokens.append(sent['tokens'])
                tok_len.append(len(sent['tokens']))

    df = pd.DataFrame({
    'file_name': file_names,
    'split': splits,
    'caption': captions,
    'tok_len': tok_len,
    'tokens': tokens
})

    print(f"Token Max Length: {df.tok_len.max()}")

    df.to_json(df_name)


def get_topk_vocab(dataset=None, topk=10000):
    files = ['data.json', 'data30.json', 'coco.json']
    
    print(files[0])


    # freqs = {}
    # tokens_list = []
    # print("asdadadsa")
    # print(files, 'sss')
    # for file in files:
    #     print(file)
    #     df = pd.read_json(file)
    #     tokens_list.extend(df.tokens.to_list()) 


    # for tokens in tqdm(tokens_list, desc='Counting Frequencies'):
    #     for word in tokens:
    #         if word not in freqs:
    #             freqs[word] = 1
    #         else:
    #             freqs[word] += 1

    # top_vocab_dict = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
    # topk_words = list(top_vocab_dict)[:topk]
    # return topk_words