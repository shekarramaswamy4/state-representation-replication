import torch 
from torch import nn
import numpy as np

from probe.probe import Probe, FSProbe
from torch.utils.data import RandomSampler, BatchSampler
from sklearn.metrics import f1_score

from collections import defaultdict

def calc_f1_score_for_labels(pred_labels, gt_labels):
    pred_detach = pred_labels.detach().numpy()
    gt_detach = gt_labels.detach().numpy()
    return f1_score(pred_detach, gt_detach)

def calculate_multiclass_accuracy(preds, labels):
    acc = float(np.sum((preds == labels).astype(int)) / len(labels))
    return acc

class ProbeHandler():
    def __init__(self, num_state_variables, encoder, state_var_mapping, is_supervised = False):
        self.num_state_variables = num_state_variables
        self.encoder = encoder
        assert self.encoder is not None # sanity
        self.is_supervised = is_supervised

        self.probes = []
        self.optimizers = []

        self.loss = nn.CrossEntropyLoss()

        self.mapping = state_var_mapping
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

    def train_epoch(self, train_episodes, train_labels, batch_size):
        '''
        Runs training for one epoch.
        '''

        tr_episodes_batched, tr_labels_batched = self.randomly_sample_for_batch(train_episodes, train_labels, batch_size)
        epoch_loss_per_state_variable = np.zeros(self.num_state_variables)
        accuracy_per_state_variable = np.zeros(self.num_state_variables)

        for ep in range(len(tr_episodes_batched)): # training for each batch
            gt_labels = tr_labels_batched[ep]
            cur_episodes = tr_episodes_batched[ep]
            for var, var_label in gt_labels.items(): # per state variable
                idx = self.mapping[var]

                cur_probe = self.probes[idx]
                cur_optim = self.optimizers[idx]
                cur_optim.zero_grad()

                # if fully supervised, FSProbe has the encoder in its init
                # if not, we use self.encoder for non FS case and make sure to stop gradient
                if self.is_supervised: 
                    pred_labels = cur_probe(cur_episodes)
                else:
                    with torch.no_grad():
                        encoder_output = self.encoder(cur_episodes)
                    pred_labels = cur_probe(encoder_output)

                var_label = torch.tensor(var_label).long()

                # loss metric
                loss = self.loss(pred_labels, var_label)
                loss_val = loss.item()
                epoch_loss_per_state_variable[idx] += loss_val

                # accuracy metric
                pred_labels = pred_labels.detach().numpy()
                pred_labels = np.argmax(pred_labels, axis=1)
                var_label = var_label.detach().numpy()

                # this is how authors compute accuracy. it is bad. we should use a different metric, eventually.
                accuracy_per_state_variable[idx] += calculate_multiclass_accuracy(pred_labels, var_label)

                loss.backward()
                cur_optim.step()
        
        accuracy_per_state_variable = np.array(accuracy_per_state_variable) / len(tr_episodes_batched)
        return epoch_loss_per_state_variable, accuracy_per_state_variable

    def run_probes(self, episodes, labels, batch_size):
        '''
        Used to determine loss / accuracy from a episodes and labels.
        (Used for both validation and testing since they are calculated
        in the same way.)
        '''
        episodes_batched, labels_batched = self.randomly_sample_for_batch(episodes, labels, batch_size)
        epoch_loss_per_state_variable = np.zeros(self.num_state_variables)
        accuracy_per_state_variable = np.zeros(self.num_state_variables)

        for ep in range(len(episodes_batched)):
            gt_labels = labels_batched[ep]
            cur_episodes = episodes_batched[ep]
            for var, var_label in gt_labels.items(): # per state variable
                idx = self.mapping[var]
                cur_probe = self.probes[idx]

                # if fully supervised, FSProbe has the encoder in its init
                # if not, we use self.encoder for non FS case. we are testing so no need to stop gradient
                if self.is_supervised: 
                    pred_labels = cur_probe(encoder_output)
                else:
                    encoder_output = self.encoder(cur_episodes)
                    pred_labels = cur_probe(encoder_output)

                var_label = torch.tensor(var_label).long()
                loss = self.loss(pred_labels, var_label)

                # TODO: f1 score
                # f1 = calc_f1_score_for_labels(pred_labels, var_label)
                # print('F1 score for Probe #' + str(idx) + ': ' + str(f1))

                # loss metric
                loss_val = loss.item()
                epoch_loss_per_state_variable[idx] += loss_val

                # accuracy metric
                pred_labels = pred_labels.detach().numpy()
                pred_labels = np.argmax(pred_labels, axis=1)
                var_label = var_label.detach().numpy()
                # this is how authors compute accuracy. it is bad. we should use a different metric, eventually.
                accuracy_per_state_variable[idx] += calculate_multiclass_accuracy(pred_labels, var_label)

        accuracy_per_state_variable = np.array(accuracy_per_state_variable) / len(episodes_batched)
        return epoch_loss_per_state_variable, accuracy_per_state_variable

    def randomly_sample_for_batch(self, episodes, labels, batch_size):
        '''
        Generates batches of length batch_size from episodes and labels.
        Output is [tensor] for data, and [state_var_dict] for labels.
        '''
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
            label_data = {}
            for i in batch:
                ep_idx, idx = self.determine_index_of_example(episode_lengths, i)
                cur_datapoint = episodes[ep_idx][idx]
                cur_label = labels[ep_idx][idx]
                batch_data.append(cur_datapoint)
                self.append_dict_to_current(label_data, cur_label)
            # turn into torch tensor
            batch_data = torch.stack(batch_data)
            my_data.append(batch_data)
            my_labels.append(label_data)
        return (my_data, my_labels)

    # returns the episode and the index of the episode that the example belongs to
    def determine_index_of_example(self, episode_lengths, index):
        '''
        Helper for randomly_sample_for_batch. Determines which episode and what
        position the current example (index) is from, dependent on non uniform
        episode_lengths.
        '''
        for i in range(0, len(episode_lengths)):
            if sum(episode_lengths[:i+1]) > index:
                return (i, index - sum(episode_lengths[:i]))
        
        # shouldn't be here
        print('ERROR: determine_index_of_example / probe_handler.py: Invalid index')
        return (0, 0)
    
    # appends newdict items to current
    def append_dict_to_current(self, current_dict, new_dict):
        if len(current_dict) == 0:
            for key, value in new_dict.items():
                current_dict[key] = [value]
        else:
            for key, value in new_dict.items():
                cur_arr = current_dict[key]
                cur_arr.append(value)
                current_dict[key] = cur_arr
        
        return current_dict

