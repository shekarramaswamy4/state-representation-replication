import torch 
from torch import nn
import numpy as np

from probe.probe import Probe, FSProbe
from torch.utils.data import RandomSampler, BatchSampler
from sklearn.metrics import f1_score

from collections import defaultdict

class appendabledict(defaultdict):
    def __init__(self, type_=list, *args, **kwargs):
        self.type_ = type_
        super().__init__(type_, *args, **kwargs)

    #     def map_(self, func):
    #         for k, v in self.items():
    #             self.__setitem__(k, func(v))

    def subslice(self, slice_):
        """indexes every value in the dict according to a specified slice
        Parameters
        ----------
        slice : int or slice type
            An indexing slice , e.g., ``slice(2, 20, 2)`` or ``2``.
        Returns
        -------
        sliced_dict : dict (not appendabledict type!)
            A dictionary with each value from this object's dictionary, but the value is sliced according to slice_
            e.g. if this dictionary has {a:[1,2,3,4], b:[5,6,7,8]}, then self.subslice(2) returns {a:3,b:7}
                 self.subslice(slice(1,3)) returns {a:[2,3], b:[6,7]}
         """
        sliced_dict = {}
        for k, v in self.items():
            sliced_dict[k] = v[slice_]
        return sliced_dict

    def append_update(self, other_dict):
        """appends current dict's values with values from other_dict
        Parameters
        ----------
        other_dict : dict
            A dictionary that you want to append to this dictionary
        Returns
        -------
        Nothing. The side effect is this dict's values change
         """
        for k, v in other_dict.items():
            self.__getitem__(k).append(v)

def calc_f1_score_for_labels(gt_labels, pred_labels):
    return f1_score(gt_labels, pred_labels)

class ProbeHandler():
    def __init__(self, num_state_variables, encoder, is_supervised = False):
        self.num_state_variables = num_state_variables
        self.encoder = encoder
        assert self.encoder is not None # sanity
        self.is_supervised = is_supervised

        self.probes = []
        self.optimizers = []

        self.loss = nn.CrossEntropyLoss()

        self.mapping = {'player_y': 0, 'enemy_y': 1, 'ball_x': 2, 'ball_y': 3, 'enemy_score': 4, 'player_score': 5}

        self.setup_probes()
    
    def setup_probes(self):
        if self.is_supervised:
            for i in range(self.num_state_variables):
                self.probes.append(FSProbe(self.encoder))
                self.optimizers.append(torch.optim.Adam(list(self.probes[i].parameters()), lr=3e-4))
        else:
            for i in range(self.num_state_variables):
                self.probes.append(Probe())
                self.optimizers.append(torch.optim.Adam(list(self.probes[i].parameters()), lr=5e-2))
        
        for i in range(self.num_state_variables):
            # TODO: add LR schedulers w warmup and cycles
            pass

    def train(self, train_episodes, train_labels, epochs = 100, batch_size = 64):
        print('--- Training Probes ---')
        self.setup_probes()

        for i in range(epochs):
            print('Epoch: ' + str(i) + ' of ' + str(epochs))
            metrics = self.train_epoch(train_episodes, train_labels, batch_size)
            print(metrics)
        print('--- Finished training probes. ---')
    
    def validate(self, val_episodes, val_labels, batch_size = 64):
        print('--- Validating Probes ---')
        metrics = self.run_probes(val_episodes, val_labels, batch_size)
        print(metrics)
        
    def test(self, test_episodes, test_labels, batch_size = 64):
        print('--- Testing Probes ---')
        metrics = self.run_probes(test_episodes, test_labels, batch_size)
        print(metrics)

    # train for one epoch
    def train_epoch(self, train_episodes, train_labels, batch_size):

        tr_episodes_batched, tr_labels_batched = self.randomly_sample_for_batch(train_episodes, train_labels, batch_size)
        epoch_loss_per_state_variable = np.zeros(self.num_state_variables)

        for ep in range(len(tr_episodes_batched)): # training for each batch
            gt_labels = tr_labels_batched[ep]
            cur_episodes = tr_episodes_batched[ep]
            for k, label in gt_labels.items(): # per state variable
                j = self.mapping[k]

                cur_probe = self.probes[j]
                cur_optim = self.optimizers[j]
                cur_optim.zero_grad()

                # if fully supervised, FSProbe has the encoder in its init
                # if not, we use self.encoder for non FS case and make sure to stop gradient
                if self.is_supervised: 
                    pred_labels = cur_probe(cur_episodes)
                else:
                    with torch.no_grad():
                        encoder_output = self.encoder(cur_episodes)
                    pred_labels = cur_probe(encoder_output)

                label = torch.tensor(label).long()
                loss = self.loss(pred_labels, label)

                loss_val = loss.item()
                epoch_loss_per_state_variable[j] += loss_val

                loss.backward()
                cur_optim.step()
        
        return epoch_loss_per_state_variable

    # currently non functional, see train_epoch
    def run_probes(self, episodes, labels, batch_size):
        '''
        Used to determine loss / accuracy from a episodes and labels.
        (Used for both validation and testing since they are calculated
        in the same way.)
        '''
        episodes_batched, labels_batched = self.randomly_sample_for_batch(episodes, labels, batch_size)
        epoch_loss_per_state_variable = np.zeros(self.num_state_variables)

        for ep in range(len(episodes_batched)):
            gt_labels = labels_batched[ep]
            cur_episodes = episodes_batched[ep]
            for j in range(self.num_state_variables):
                cur_probe = self.probes[j]

                # if fully supervised, FSProbe has the encoder in its init
                # if not, we use self.encoder for non FS case. we are testing so no need to stop gradient
                if self.is_supervised: 
                    pred_labels = cur_probe.forward(encoder_output)
                else:
                    encoder_output = self.encoder(cur_episodes)
                    pred_labels = cur_probe.forward(encoder_output)

                loss = self.loss(gt_labels, pred_labels)

                f1 = calc_f1_score_for_labels(gt_labels, pred_labels)
                print('F1 score for Probe #' + str(j) + ': ' + str(f1))

                epoch_loss_per_state_variable[j] += loss
                # additional metrics

        return epoch_loss_per_state_variable

    def randomly_sample_for_batch(self, episodes, labels, batch_size):
        episode_lengths = []
        for ep in episodes:
            episode_lengths.append(len(ep))
        frames_count = sum(episode_lengths)

        # batches is [[batch_size]]
        batches = BatchSampler(RandomSampler(range(frames_count), replacement=False), batch_size, drop_last=True)

        my_data = []
        my_labels = []
        for batch in batches:
            batch_data = []
            label_data = appendabledict()
            for i in batch:
                ep_idx, idx = self.determine_index_of_example(episode_lengths, i)
                cur_datapoint = episodes[ep_idx][idx]
                cur_label = labels[ep_idx][idx]

                batch_data.append(cur_datapoint)
                label_data.append_update(cur_label)
            # turn into torch tensor
            batch_data = torch.stack(batch_data)
            my_data.append(batch_data)
            my_labels.append(label_data)
        # turn into torch tensor
        # my_data = torch.stack(my_data)
        # my_labels = torch.stack(my_labels)
        return (my_data, my_labels)

    # returns the episode and the index of the episode that the example belongs to
    def determine_index_of_example(self, episode_lengths, index):
        for i in range(0, len(episode_lengths)):
            if sum(episode_lengths[:i+1]) > index:
                return (i, index - sum(episode_lengths[:i]))
        
        # shouldn't be here
        print('ERROR: determine_index_of_example / probe_handler.py: Invalid index')
        return (0, 0)