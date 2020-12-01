import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from models import Encoder,Decoder,Attention,Seq2Seq
from constants import PAD_TOKEN,EOS_TOKEN,UNK_TOKEN,BOS_TOKEN


def load_model(SRC,TRG):

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

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

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
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
