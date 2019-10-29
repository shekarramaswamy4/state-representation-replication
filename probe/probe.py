import torch
from torch import nn

class Probe(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer = nn.Linear(256, 255) # still don't understand why this is 255

	def forward(self, x):
		return self.layer(x)
