import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchtext.data.metrics import bleu_score
import torch.optim as optim

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np

from datasets import Vocabulary

from tqdm import tqdm


# transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def caption_image(image, model, vocab, max_len=50):

    model.eval()
    result_caption = []

    with torch.no_grad():
        context = model.encoder(image.to(model.device)).unsqueeze(0)
        #hidden = torch.tensor(vocab.stoi["<sos>"]).unsqueeze(0).to(model.device)
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


def predict_test(test_dict, imgs_path, model, vocab, max_len=50, n_images=100):

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


def print_scores(preds, trgs):
    print("1:", bleu_score(preds, trgs, max_n=1, weights=[1])*100)
    print("2:", bleu_score(preds, trgs, max_n=2, weights=[.5, .5])*100)
    print("3:", bleu_score(preds, trgs, max_n=3, weights=[.33, .33, .33])*100)
    print("4:", bleu_score(preds, trgs)*100)

    
    

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for idx, (imgs, captions) in tqdm(enumerate(iterator), total=len(iterator), leave=False, desc="training"):
        
        optimizer.zero_grad()
        
        imgs = imgs.to(model.device)
        captions = captions.to(model.device)
        
        outputs = model(imgs, captions[:-1])
        #output = [trg len, batch size, output dim]
        loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
        

        loss.backward()
        
        # clip the grads
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (images, captions) in tqdm(enumerate(iterator), total=len(iterator), leave=False, desc="Evaluating"):
            
            images = images.to(model.device)
            captions = captions.to(model.device)
            
            outputs = model(images, captions[:-1])
            #output = [trg len, batch size, output dim]
            
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs