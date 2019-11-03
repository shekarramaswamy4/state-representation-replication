import torch 
from torch import nn

from probe import Probe, FSProbe

from torch.utils.data import RandomSampler, BatchSampler

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
				self.optimizers.append(torch.optim.Adam(list(self.probes[k].parameters()), lr=3e-4))
		else:
			for i in range(self.num_state_variables):
				self.probes.append(Probe())
				self.optimizers.append(torch.optim.Adam(list(self.probes[k].parameters()), lr=5e-2))
		
		for i in range(self.num_state_variables):
		
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
	
	def randomly_sample_for_batch(self, episodes, labels, batch_size):
		episode_lengths = []
		for ep in episodes:
			episode_lengths.append(len(ep))
		frames_count = sum(episode_lengths)

		# batches is [[batch_size]]
        batches = BatchSampler(RandomSampler(frames_count), replacement=False, num_samples=frames_count), batch_size, drop_last=True)

		my_data = []
		my_labels = []
		for batch in batches:
			batch_data = []
			label_data = []
			for i in batch:
				ep_idx, idx = self.determine_index_of_example(episode_lengths, i)
				cur_datapoint = episodes[ep_idx][idx]
				cur_label = labels[ep_idx][idx]

				batch_data.append(cur_datapoint)
				label_data.append(cur_label)
			my_data.append(batch_data)
			my_labels.append(label_data)
		
		return (my_data, my_labels)

	# returns the episode and the index of the episode that the example belongs to
	def determine_index_of_example(self, episode_lengths, index):
		for i in range(1, episode_lengths + 1):
			if sum(episode_lengths[:i]) >= index:
				return (i, index - sum(episode_lengths[:i-1]))
		
		# shouldn't be here
		print('ERROR: determine_index_of_example / probe_handler.py: Invalid index')
		return (0, 0)

	# used to determine validation and testing loss / accuracy
	def test_probes(self, episodes, labels):
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
						test_episodes, test_labels, epochs = 100, batch_size = 64):
		print('--- Training Probes ---')
		self.setup_probes()

		for i in range(epochs):
			print('Epoch: ' + str(i) + ' of ' + str(epochs))
			self.train_epoch(train_episodes, train_labels)
		
		print('Finished training probes.')
        