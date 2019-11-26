import torch
import torchvision
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, BatchSampler
from encoders.rand_cnn import RandCNN

import matplotlib.pyplot as plt

class StDimHandler:
    '''
    Trains an encoder based on the InfoNCE ST-DIM method.
    '''

    def __init__(self):
        self.encoder = RandCNN()
        # TODO: using the bilinear layers doesn't allow for a batch size change
        # TODO: the 64 here is set as the default batch size, but this might change, probably why they used a matmul instead 
        self.bilinear_gl = nn.Bilinear(256, 128, 64, bias=False)
        self.bilinear_ll = nn.Bilinear(128, 128, 64, bias=False)
        
        # TODO: puts these into "train" mode, unclear what effect this has but the authors did it
        self.encoder.train(), self.bilinear_gl.train(), self.bilinear_ll.train()
        
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                        list(self.bilinear_gl.parameters()) +
                                        list(self.bilinear_ll.parameters()),
                                        lr=3e-4, eps=1e-5)

    def generate_batches(self, episodes, batch_size):
        '''
        Generates batches of episodes for sequential frames.
        :param episodes: 2D np array [num_episodes x episode_length]
                         each element is a 210 x 160 frame
        :return: batches of subsequent frames, dimensions: num_batches x batch_size,
                 each batch looks like [(first_frame_1,  next_frame_1), (first_frame_2,  next_frame_2), ...]
        '''
        episode_lengths = []
        for ep in episodes:
            episode_lengths.append(len(ep))
        frames_count = sum(episode_lengths)

        frame_batches = []
        # do this twice to approximately cover all the frames because the random sampler only runs through half the indices
        # don't want any frames that are right next to each other, sample for indices that are half the number of frames
        frame_idx_batches = BatchSampler(RandomSampler(range(frames_count), replacement=False), batch_size, drop_last=True)
        for frame_idx_batch in frame_idx_batches:
            frame_batches.append([])
            x = []
            # multiply all indices by 2 to account for previous sampling scheme so every index is at least separated by 1
            frame_idx_batch = [2 * i for i in frame_idx_batch]
            for frame_idx in frame_idx_batch:
                ep_idx, idx_within_ep = self.determine_index_of_example(episode_lengths, frame_idx)
                # if it was the first frame in the episode, take the frame after
                if idx_within_ep == 0:
                    # frame_batches[-1].append(torch.stack((episodes[ep_idx][idx_within_ep] / 255.0, episodes[ep_idx][idx_within_ep + 1] / 255.0)))
                    frame_batches[-1].append((frame_idx, frame_idx + 1))
                    x.append((ep_idx,idx_within_ep+1))
                else:
                    # frame_batches[-1].append(torch.stack((episodes[ep_idx][idx_within_ep - 1] / 255.0, episodes[ep_idx][idx_within_ep] / 255.0)))
                    frame_batches[-1].append((frame_idx-1, frame_idx))
                    x.append((ep_idx,idx_within_ep))
            # make into a Tensor
            # frame_batches[-1] = torch.stack(frame_batches[-1])
            z = list(zip(*x))
            x, y = z[0], z[1]
            plt.scatter(x, y)
            plt.show()
        return frame_batches

    def determine_index_of_example(self, episode_lengths, index):
        '''
        Helper for generate_batch. Determines which episode and what
        position the current example (index) is from, dependent on non uniform
        episode_lengths.
        
        :returns: ep_idx, idx
        '''
        for i in range(0, len(episode_lengths)):
            if sum(episode_lengths[:i+1]) > index:
                return (i, index - sum(episode_lengths[:i]))
        
        # shouldn't be here
        print('ERROR: determine_index_of_example / probe_handler.py: Invalid index')
        return (0, 0)
        
    def train(self, train_episodes, train_labels, val_episodes=None, val_labels=None, epochs=100, batch_size=64):
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
            
            frame_batches = self.generate_batches(train_episodes, batch_size)
            
            metrics = self.train_epoch(frame_batches)
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

    def train_epoch(self, frame_batches):
        '''
        Runs training for one epoch.
        :param frame_batches: list of batches of paired frames, dimensions: num_batches x batch_size x 2 x (1, 210, 160)
        '''
        # training for each batch
        i = 1
        for frame_batch in frame_batches:
            print(f"batch {i}")
            i+=1
            # frame_batch dimensions: batch_size x 2 x (1, 210, 160)
            global_local_loss = 0
            local_local_loss = 0
            
            # frames dimensions: batch_size x 1 x (1, 210, 160)
            first_frames = frame_batch[:, 0]
            second_frames = frame_batch[:, 1]
            
            # Get encoder outputs from RandCNN
            # permute them to: batch_size x 1 x height x width x channel (helps with bilinear layer computation)
            first_frames_global = self.encoder(first_frames, intermediate_layer=False)
            second_frames_local = self.encoder(second_frames, intermediate_layer=True).permute(0, 2, 3, 1)
            
            # global-local loss
            # indicate how many patches to count
            second_height_range = second_frames_local.shape[1]
            second_width_range = second_frames_local.shape[2]
            
            # loop through local patches of next frames
            batch_size = frame_batch.shape[0]
            for h in range(second_height_range):
                for w in range(second_width_range):
                    g_mn = self.bilinear_gl(first_frames_global, second_frames_local[:, h, w, :])
                    # TODO: this cross entropy loss highly optimized 
                    loss_for_patch = F.cross_entropy(g_mn, torch.arange(batch_size))
                    global_local_loss += loss_for_patch
            global_local_loss /= second_height_range * second_width_range
            print(f"global_local_loss: {global_local_loss}")
            
            # local-local loss
            # TODO: this is redundant computation for the encoder
            first_frames_local = self.encoder(first_frames, intermediate_layer=True).permute(0, 2, 3, 1)
            
            # loop through local patches of next frames
            batch_size = frame_batch.shape[0]
            for h in range(second_height_range):
                for w in range(second_width_range):
                    g_mn = self.bilinear_ll(first_frames_local[:, h, w, :], second_frames_local[:, h, w, :])
                    # TODO: this cross entropy loss highly optimized 
                    loss_for_patch = F.cross_entropy(g_mn, torch.arange(batch_size))
                    local_local_loss += loss_for_patch
            local_local_loss /= second_height_range * second_width_range
            print(f"local_local_loss: {local_local_loss}")
            
            total_loss = global_local_loss + local_local_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        #         cur_probe = self.probes[idx]
        #         cur_optim = self.optimizers[idx]
        #         cur_optim.zero_grad()

        #         # if fully supervised, FSProbe has the encoder in its init
        #         # if not, we use self.encoder for non FS case and make sure to stop gradient
        #         if self.is_supervised:
        #             pred_labels = cur_probe(cur_episodes)
        #         else:
        #             with torch.no_grad():
        #                 encoder_output = self.encoder(cur_episodes)
        #             pred_labels = cur_probe(encoder_output)

        #         var_label = torch.tensor(var_label).long()

        #         # loss metric
        #         loss = self.loss(pred_labels, var_label)
        #         loss_val = loss.item()
        #         epoch_loss_per_state_variable[idx] += loss_val

        #         # accuracy metric
        #         pred_labels = pred_labels.detach().numpy()
        #         pred_labels = np.argmax(pred_labels, axis=1)
        #         var_label = var_label.detach().numpy()

        #         # this is how authors compute accuracy. it is bad. we should use a different metric, eventually.
        #         accuracy_per_state_variable[idx] += calculate_multiclass_accuracy(
        #             pred_labels, var_label)

        #         loss.backward()
        #         cur_optim.step()

        # epoch_loss_per_state_variable = np.array(
        #     epoch_loss_per_state_variable) / len(tr_episodes_batched)
        # accuracy_per_state_variable = np.array(
        #     accuracy_per_state_variable) / len(tr_episodes_batched)
        # return epoch_loss_per_state_variable, accuracy_per_state_variable
