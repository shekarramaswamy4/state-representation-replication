import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

class StDimHandler:
    '''
    Trains an encoder based on the InfoNCE ST-DIM method.
    '''
    def __init__(self, encoder):
        self.encoder = encoder

    def generate_batch(self, episodes):
        '''
        Generates batch of episodes for sequential frames.
        :param episodes: 2D np array [num_episodes x episode_length]
                         each element is a 210 x 160 frame
        :return: generator which returns two sequential frames, x_t and x_tprev
        '''
        pass

    def train(self, train_episodes, train_labels, val_episodes = None, val_labels = None, epochs = 100, batch_size = 64):
        '''
        Trains the encoder based on the global-local and local-local objectives.
        '''
        print('--- Training ---')

        train_loss_arr = []
        train_acc_arr = []

        val_loss_arr = []
        val_acc_arr = []
        val_f1_arr = []
        
        for i in range(epochs):
            print('Epoch: ' + str(i + 1) + ' of ' + str(epochs))
            metrics = self.train_epoch(train_episodes, train_labels, batch_size)
            train_loss_arr.append(metrics[0])
            train_acc_arr.append(metrics[1])
            
            if val_episodes is not None and val_labels is not None:
                print()
                print('Validation: ' + str(i + 1) + ' of ' + str(epochs))
                metrics = self.run_probes(val_episodes, val_labels, batch_size)
                self.print_metrics(metrics)
                val_loss_arr.append(metrics[0])
                val_acc_arr.append(metrics[1])
                val_f1_arr.append(metrics[2])
                print()
                print()
        

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
            # TODO: this depends on dictionary preserving game state variable order
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
    
        epoch_loss_per_state_variable = np.array(epoch_loss_per_state_variable) / len(tr_episodes_batched)
        accuracy_per_state_variable = np.array(accuracy_per_state_variable) / len(tr_episodes_batched)
        return epoch_loss_per_state_variable, accuracy_per_state_variable
