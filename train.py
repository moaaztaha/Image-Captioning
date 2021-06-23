import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import Encoder, DecoderWithAttention
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# training parameters
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 100  # print training/validation stats every __ batches



def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, logger=None):
    # performs one epoch's training

    
    encoder.train()
    decoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # move to gpu, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.unsqueeze(1).to(device)

        # forward prop
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # get the words after <sos>
        targets = caps_sorted[:, 1:]

        # remove timesteps that we didn't decode at or are pads
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # calculate the loss
        loss = criterion(scores, targets)

        # doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # back prop
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        
        # keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()


        # print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
            
        # tensorboard 
        logger.add_scalar(f'Loss/train/{str(epoch)}', losses.val, i)
        logger.add_scalar(f'top5acc/train/{str(epoch)}', top5accs.val, i)
        
        

def validate(val_loader, encoder, decoder, criterion, vocab, epoch=None, logger=None):

    decoder.eval()
    if encoder is not None:
        encoder.eval()

    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list() # true captions for calculating the bleu scores
    hypotheses = list() # hypotheses (predictions)

    with torch.no_grad():
        # batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.unsqueeze(1).to(device)


            # forward prop
            if encoder is not None:
                imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # calculate loss
            loss = criterion(scores, targets)

            # doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))
            
            # tensorboard 
            logger.add_scalar(f'Loss/valid/{str(epoch)}', losses.val, i)
            logger.add_scalar(f'top5acc/valid/{str(epoch)}', top5accs.val, i)





            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                references.append(vocab.indextostring(img_caps))
            

            # hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]]) # remove pads
            
            preds = vocab.indextostring(temp_preds)
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)
            
            # debug
            # if i % 50 == 0 and i != 0:
            # break

        # debug
        # return references, hypotheses
        # Calculate BLEU-4 scores
        #bleu4 = corpus_bleu(references, hypotheses)

        # print scores
        b1, b2, b3, b4, m = print_scores(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=b4))
        
    return b1, b2, b3, b4, m



def fit(t_params, checkpoint=None, m_params=None, logger=None):

    # info
    data_name = t_params['data_name']
    imgs_path = t_params['imgs_path']
    df_path = t_params['df_path']
    vocab = t_params['vocab']

    start_epoch = 0
    epochs_since_improvement = 0
    best_bleu4 = 0
    epochs = t_params['epochs']
    batch_size = t_params['batch_size']
    workers = t_params['workers']
    encoder_lr = t_params['encoder_lr']
    decoder_lr = t_params['decoder_lr']
    fine_tune_encoder = t_params['fine_tune_encoder']


    # init / load checkpoint
    if checkpoint is None:

        # getting hyperparameters
        attention_dim = m_params['attention_dim']
        embed_dim = m_params['embed_dim']
        decoder_dim = m_params['decoder_dim']
        encoder_dim = m_params['encoder_dim']
        dropout = m_params['dropout']

        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                      embed_dim=embed_dim,
                                      decoder_dim=decoder_dim,
                                      encoder_dim=encoder_dim,
                                      vocab_size=len(vocab),
                                      dropout=dropout)
        decoder_optimizer = torch.optim.RMSprop(params=filter(lambda p:p.requires_grad, decoder.parameters()),
                                            lr=decoder_lr)
        
        decoder_scheduler = ReduceLROnPlateau(decoder_optimizer)


        encoder=Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.RMSprop(params=filter(lambda p:p.requires_grad, encoder.parameters()),
                                            lr=encoder_lr) if fine_tune_encoder else None
        if fine_tune_encoder:
            encoder_scheduler = ReduceLROnPlateau(encoder_optimizer)
        
    # load checkpoint
    else:
        checkpoint = torch.load(checkpoint)
        print('Loaded Checkpoint!!')
        start_epoch = checkpoint['epoch'] + 1
        print(f"Starting Epoch: {start_epoch}")
        epochs_since_improvement = checkpoint['epochs_since_imrovment']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['deocder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.RMSprop(params=filter(lambda p:p.requires_grad, encoder.parameters()),
                                                lr=encoder_lr)
    # move to gpu, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    
    # loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # dataloaders
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print('Loading Data')
    train_loader, val_loader = get_loaders(batch_size, imgs_path, df_path, transform, vocab, False ,workers)
    print('_'*50)

    print('-'*20, 'Fitting', '-'*20)
    for epoch in range(start_epoch, epochs):

        # if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
        #     adjust_learning_rate(decoder_optimizer, 0.8)
        #     if fine_tune_encoder:
        #         adjust_learning_rate(encoder_optimizer, 0.8)
        
        print('_'*50)
        print('-'*20, 'Training', '-'*20)
        # one epoch of training
        epoch_time = AverageMeter()
        start_time = time.time()
        train(train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
            logger=logger)
        epoch_time.update(time.time() - start_time)
        print(f"Epoch train time {epoch_time.val:.3f} (epoch_time.avg:.3f)")

        # one epoch of validation
        epoch_time = AverageMeter()
        start_time = time.time()
        print('-'*20, 'Validation', '-'*20)
        b1, b2, b3, recent_bleu4, m = validate(val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            vocab=vocab,
            epoch=epoch,
            logger=logger)
        epoch_time.update(time.time() - start_time)
        # tensorboard 
        logger.add_scalar(f'b-1/valid', b1, epoch)
        logger.add_scalar(f'b-2/valid', b2, epoch)
        logger.add_scalar(f'b-3/valid', b3, epoch)
        logger.add_scalar(f'b-4/valid', recent_bleu4, epoch)
        logger.add_scalar(f'Meteor/valid', m, epoch)
        print(f"Epoch validation time {epoch_time.val:.3f} (epoch_time.avg:.3f)")

        
        # check for improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print(f'\nEpochs since last improvement: {epochs_since_improvement,}')
        else:
            # reset
            epochs_since_improvement = 0
        

        # stop training if no improvement for 5 epochs
        if epochs_since_improvement == 5:
            print('No improvement for 3 consecutive epochs, terminating...')
            break
        
        # learning rate schedular
        decoder_scheduler.step(recent_bleu4)
        if fine_tune_encoder:
            encoder_scheduler.step(recent_bleu4)

        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
            decoder_optimizer, recent_bleu4, is_best)