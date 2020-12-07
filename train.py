import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torchtext.data import BucketIterator
from models import Encoder,Decoder,Seq2Seq
from constants import PAD_TOKEN,EOS_TOKEN,UNK_TOKEN,BOS_TOKEN, CLIP
from tqdm import tqdm
import os
import pdb

cwd = os.getcwd()

def load_model(SRC,TRG):

    device = torch.device('cpu')

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    # ENC_EMB_DIM = 256
    # DEC_EMB_DIM = 256
    # ENC_HID_DIM = 512
    # DEC_HID_DIM = 512
    # ATTN_DIM = 64
    # ENC_DROPOUT = 0.5
    # DEC_DROPOUT = 0.5

    ENC_EMB_DIM = 32
    DEC_EMB_DIM = 32
    ENC_HID_DIM = 64
    DEC_HID_DIM = 64
    ATTN_DIM = 8
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)

    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)


    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())


    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'The model has {count_parameters(model):,} trainable parameters')


    PAD_IDX = TRG.vocab.stoi['PAD_TOKEN']

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    return model,optimizer,criterion


def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module):

    model.train()

    for epoch in range(1):

        epoch_loss = 0

        for _, batch in tqdm(enumerate(iter(iterator))):

            src = batch.src
            trg = batch.trg

            #print(src[0].shape, trg[0].shape)

            optimizer.zero_grad()

            output = model(src, trg[0])

            output = output[1:].view(-1, output.shape[-1])

            trg2 = trg[0][:,1:].reshape(output.shape[0])
            
            #pdb.set_trace()

            # output should be 2-dimensional, and trg2 should be 1-dimensional.
            # first dimension of output should match dimension of trg2.

            loss = criterion(output, trg2)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

            optimizer.step()

            epoch_loss += loss.item()

        print(epoch_loss/len(iterator))

        torch.save(model.state_dict(), os.path.join(cwd, 'pretrained', 'model_1_'+str(epoch)+'.pt'))

    return model
