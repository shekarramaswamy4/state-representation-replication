import torch
from torch import nn

class Probe(nn.Module):
	def __init__(self, input_features = 256, output_features = 255):
		super().__init__()
		self.layer = nn.Linear(input_features, output_features)

	def forward(self, x):
		return self.layer(x)

class FSProbe(nn.Module):
	def __init__(self, encoder, input_features = 256, output_features = 255):
		super().__init__()
		self.encoder = encoder
		self.layer = nn.Linear(input_features, output_features)

	def forward(self, x):
		return self.layer(self.encoder(x))