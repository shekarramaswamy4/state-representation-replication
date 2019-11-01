import torch 
from torch import nn

from probe import Probe, FSProbe

class ProbeHandler():
	def __init__(self, num_state_variables, encoder = None, is_supervised = False):
		self.num_state_variables = num_state_variables
		self.is_supervised = is_supervised
		self.encoder = encoder

		self.probes = []
		self.optimizers = []

		self.loss = nn.CrossEntropyLoss()

		self.setup_probes()
	
	def setup_probes(self):
		if self.is_supervised:
			for i in range(self.num_state_variables):
				self.probes.append(FSProbe())
		else:
			for i in range(self.num_state_variables):
				self.probes.append(Probe())
		
		for i in range(self.num_state_variables):
			self.optimizers.append(torch.optim.Adam(list(self.probes[k].parameters()), lr=3e-4)) # Karpathy's constant
		
		# TODO: add LR schedulers w warmup and cycles
	
	def train_epoch(self, train_episodes, train_labels):
		# TODO: randomize input, outer loop should iterate through some subset of episodes/labels
		epoch_loss_per_state_variable = np.zeros(self.num_state_variables)

		for j in range(self.num_state_variables):
			cur_probe = self.probes[j]
			cur_optim = self.optimizers[j]
			cur_optim.zero_grad()

			gt_label = ''
			pred_label = cur_probe.forward(datapoint) # TODO

			loss = self.loss(gt_label, pred_label)

			epoch_loss_per_state_variable[k] += loss

			loss.backward()
			cur_optim.step()
		
		# log epoch loss for each state variable

	# used to determine validation and testing loss / accuracy
	def test_probes(episodes, labels):
		epoch_loss_per_state_variable = np.zeros(self.num_state_variables)

		for j in range(self.num_state_variables):
			cur_probe = self.probes[j]

			gt_label = ''
			pred_label = cur_probe.forward(datapoint) # TODO

			loss = self.loss(gt_label, pred_label)

			epoch_loss_per_state_variable[k] += loss

		# log test loss for each state variable	


	def train(self, train_episodes, train_labels, \
						val_episodes, val_labels, \ 
						test_episodes, test_labels, epochs = 100):
		print('--- Training Probes ---')
		self.setup_probes()

		for i in range(epochs):
			print('Epoch: ' + str(i) + ' of ' + str(epochs))
			self.train_epoch(train_episodes, train_labels)
		
		print('Finished training probes.')
        