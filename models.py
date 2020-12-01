import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import pdb


class Encoder(nn.Module):
	def __init__(self,
				 input_dim: int,
				 emb_dim: int,
				 enc_hid_dim: int,
				 dec_hid_dim: int,
				 dropout: float):
		super().__init__()

		self.input_dim = input_dim
		self.emb_dim = emb_dim
		self.enc_hid_dim = enc_hid_dim
		self.dec_hid_dim = dec_hid_dim
		self.dropout = dropout

		self.embedding = nn.Embedding(input_dim, emb_dim)

		self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional = False, num_layers = 2)

		self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self,
				src: Tensor) -> Tuple[Tensor]:

		#pdb.set_trace()

		embedded = self.dropout(self.embedding(src))

		outputs, hidden = self.rnn(embedded)

		hidden = torch.tanh(self.fc(torch.cat((hidden[-1][-2,:,:], hidden[-1][-1,:,:]), dim = 1)))

		return outputs, hidden

class Decoder(nn.Module):
	def __init__(self,
				 output_dim: int,
				 emb_dim: int,
				 enc_hid_dim: int,
				 dec_hid_dim: int,
				 dropout: int):
		super().__init__()

		self.emb_dim = emb_dim
		self.enc_hid_dim = enc_hid_dim
		self.dec_hid_dim = dec_hid_dim
		self.output_dim = output_dim
		self.dropout = dropout

		self.embedding = nn.Embedding(output_dim, emb_dim)

		self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

		self.out = nn.Linear(emb_dim, output_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self,
				decoding_input: Tensor,
				decoder_hidden: Tensor,
				encoder_outputs: Tensor) -> Tuple[Tensor]:

		decoding_input = decoding_input.unsqueeze(0)

		embedded = self.dropout(self.embedding(decoding_input))

		

		output, decoder_hidden = self.rnn(embedded, decoder_hidden.unsqueeze(0))

		embedded = embedded.squeeze(0)
		output = output.squeeze(0)
		decoding_input = decoding_input.squeeze(0)

		output = self.out(torch.cat((output,
									 decoding_input,
									 embedded), dim = 1))

		return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
	def __init__(self,
				 encoder: nn.Module,
				 decoder: nn.Module,
				 device: torch.device):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.device = device

	def forward(self,
				src: Tensor,
				trg: Tensor,
				teacher_forcing_ratio: float = 0.5) -> Tensor:

		batch_size = src.shape[1]
		max_len = trg.shape[0]
		trg_vocab_size = self.decoder.output_dim

		outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

		encoder_outputs, hidden = self.encoder(src)

		# first input to the decoder is the <sos> token
		output = trg[0,:]

		for t in range(1, max_len):
			output, hidden = self.decoder(output, hidden, encoder_outputs)
			outputs[t] = output
			teacher_force = random.random() < teacher_forcing_ratio
			top1 = output.max(1)[1]
			output = (trg[t] if teacher_force else top1)

		return outputs