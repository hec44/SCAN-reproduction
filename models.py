import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import pdb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
				src: tuple) -> Tuple[Tensor]:

		embedded = self.dropout(self.embedding(src[0]))

		packed = pack_padded_sequence(embedded, src[1].cpu(),
                                      batch_first=True)

		outputs, (hidden, memory) = self.rnn(packed)

		outputs, _ = pad_packed_sequence(outputs, batch_first=True)

		#hidden = torch.tanh(self.fc(torch.cat((hidden[-1][-2,:,:], hidden[-1][-1,:,:]), dim = 1)))

		# returning the last state of the hidden layer
		return outputs, hidden, memory

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

		self.rnn = nn.LSTM(input_size = emb_dim, hidden_size = dec_hid_dim, num_layers = 2)

		self.out = nn.Linear(dec_hid_dim, output_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self,
				decoding_input: Tensor,
				decoder_hidden: Tensor,
				decoder_memory: Tensor) -> Tuple[Tensor]:

		decoding_input = decoding_input.unsqueeze(0)

		embedded = self.dropout(self.embedding(decoding_input))

		#output, (decoder_hidden, decoder_memory) = self.rnn(embedded, decoder_hidden.unsqueeze(0))
		output, (decoder_hidden, decoder_memory) = self.rnn(embedded, (decoder_hidden, decoder_memory))

		#print('\n\n\n\nhehelolz\n\n\n\n')

		embedded = embedded.squeeze(0)
		output = output.squeeze(0)
		decoding_input = decoding_input.squeeze(0)

		#output = self.out(torch.cat((output, decoding_input, embedded), dim = 1.0))
		output = self.out(output)

		return output, decoder_hidden, decoder_memory


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
				src: tuple,
				trg: Tensor,
				teacher_forcing_ratio: float = 0.5) -> Tensor:

		batch_size = src[0].shape[0]
		max_len = trg.shape[1]
		trg_vocab_size = self.decoder.output_dim

		#outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
		outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

		_, hidden, memory = self.encoder(src)

		# first input_vec to the decoder is the <sos> token
		#output = trg[:,0]
		input_vec = trg[:,0]

		for t in range(1, max_len):
			output, hidden, memory = self.decoder(input_vec, hidden, memory)
			outputs[t] = output
			teacher_force = random.random() < teacher_forcing_ratio
			top1 = output.argmax(1)
			input_vec = (trg[:,t] if teacher_force else top1)

		return outputs