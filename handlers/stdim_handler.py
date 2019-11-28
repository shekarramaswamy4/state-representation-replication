import torch
import torchvision
import numpy as np
import gc

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, BatchSampler
from encoders.rand_cnn import RandCNN

import matplotlib.pyplot as plt

old_implementation = False

class StDimHandler:
    '''
    Trains an encoder based on the InfoNCE ST-DIM method.
    '''

    def __init__(self):
        self.encoder = RandCNN()
        # TODO: using the bilinear layers doesn't allow for a batch size change
        # TODO: the 64 here is set as the default batch size, but this might change, probably why they used a matmul instead 
        self.bilinear_gl = nn.Bilinear(256, 128, 64)
        self.bilinear_ll = nn.Bilinear(128, 128, 64)
        self.encoder.train(), self.bilinear_gl.train(), self.bilinear_ll.train()
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                        list(self.bilinear_gl.parameters()) +
                                        list(self.bilinear_ll.parameters()),
                                        lr=3e-4, eps=1e-5)
        
        if old_implementation:
            self.linear_gl = nn.Linear(256, 128)  # x1 = global, x2=patch, n_channels = 32
            self.linear_ll = nn.Linear(128, 128)
            # TODO: puts these into "train" mode, unclear what effect this has but the authors did it
            self.encoder.train(), self.linear_gl.train(), self.linear_ll.train()
            self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                            list(self.linear_gl.parameters()) +
                                            list(self.linear_ll.parameters()),
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

        # frame_batches = []
        frame_idx_batches = BatchSampler(RandomSampler(range(frames_count), replacement=False), batch_size, drop_last=True)
        
        for frame_idx_batch in frame_idx_batches:
            # frame_batches.append([])
            cur_frame_batch = []
            for frame_idx in frame_idx_batch:
                ep_idx, idx_within_ep = self.determine_index_of_example(episode_lengths, frame_idx)
                # if it was the first frame in the episode, take the frame after
                if idx_within_ep == 0:
                    # frame_batches[-1].append(torch.stack((episodes[ep_idx][idx_within_ep] / 255.0, episodes[ep_idx][idx_within_ep + 1] / 255.0)))
                    cur_frame_batch.append(torch.stack((episodes[ep_idx][idx_within_ep] / 255.0, episodes[ep_idx][idx_within_ep + 1] / 255.0)))
                else:
                    # frame_batches[-1].append(torch.stack((episodes[ep_idx][idx_within_ep - 1] / 255.0, episodes[ep_idx][idx_within_ep] / 255.0)))
                    cur_frame_batch.append(torch.stack((episodes[ep_idx][idx_within_ep - 1] / 255.0, episodes[ep_idx][idx_within_ep] / 255.0)))
            # make into a tensor
            # frame_batches[-1] = torch.stack(frame_batches[-1])
            yield torch.stack(cur_frame_batch)
        # return frame_batches

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
        
        print("loading saved models")
        if old_implementation:
            self.encoder.load_state_dict(torch.load("encoders/STDIM-RandCNN"))
            self.linear_gl.load_state_dict(torch.load("encoders/STDIM-linear_gl"))
            self.linear_ll.load_state_dict(torch.load("encoders/STDIM-linear_ll"))

        avg_train_gl_loss = []
        avg_train_ll_loss = []

        avg_val_gl_loss = []
        avg_val_ll_loss = []

        for i in range(epochs):
            print('Epoch: ' + str(i + 1) + ' of ' + str(epochs))
            
            train_frame_batches = self.generate_batches(train_episodes, batch_size)
            avg_gl_loss, avg_ll_loss = self.train_epoch(train_frame_batches)
            del train_frame_batches
            gc.collect()
            
            print(f"avg_train_gl_loss: {avg_gl_loss}")
            print(f"avg_train_ll_loss: {avg_ll_loss}")
            
            avg_train_gl_loss.append(avg_gl_loss)
            avg_train_ll_loss.append(avg_ll_loss)

            if val_episodes is not None and val_labels is not None:
                print()
                print('Validation: ' + str(i + 1) + ' of ' + str(epochs))
                
                validation_frame_batches = self.generate_batches(val_episodes, batch_size)
                
                avg_gl_loss, avg_ll_loss = self.train_epoch(validation_frame_batches, train_mode=False)
                
                print(f"avg_val_gl_loss: {avg_gl_loss}")
                print(f"avg_val_ll_loss: {avg_ll_loss}")
                
                avg_val_gl_loss.append(avg_gl_loss)
                avg_val_ll_loss.append(avg_ll_loss)
                
                del validation_frame_batches
                gc.collect()

            # save encoder
            print("Saving ST-DIM trained encoder...")
            torch.save(self.encoder.state_dict(), "encoders/STDIM-RandCNN")
            torch.save(self.linear_gl.state_dict(), "encoders/STDIM-linear_gl")
            torch.save(self.linear_ll.state_dict(), "encoders/STDIM-linear_ll")
            
        print("Average train global-local losses:")
        print(avg_train_gl_loss)
        print("Average train local-local losses:")
        print(avg_train_ll_loss)
        print("Average validation global-local losses:")
        print(avg_val_gl_loss)
        print("Average validation local-local losses:")
        print(avg_val_ll_loss)
    def train_epoch(self, frame_batches, train_mode = True):
        '''
        Runs training for one epoch.
        :param frame_batches: list of batches of paired frames, dimensions: num_batches x batch_size x 2 x (1, 210, 160)
        :return: average global-local loss for this epoch, average local-local loss for this epoch
        '''
        epoch_gl_avg_loss = 0
        epoch_ll_avg_loss = 0
        
        # training for each batch
        for batch_index, frame_batch in enumerate(frame_batches):
            # frame_batch dimensions: batch_size x 2 x (1, 210, 160)
            global_local_loss = 0
            local_local_loss = 0
            
            # frames dimensions: batch_size x 1 x (1, 210, 160)
            first_frames = frame_batch[:, 0]
            second_frames = frame_batch[:, 1]
            
            # Get encoder outputs from RandCNN
            # permute them to: batch_size x 1 x height x width x channel for matrix multiplications
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
                    if old_implementation:
                        global_times_W = self.linear_gl(first_frames_global)
                        g_mn = torch.matmul(global_times_W, second_frames_local[:, h, w, :].t())
                        loss_for_patch = F.cross_entropy(g_mn, torch.arange(batch_size))
                    else:
                        g_mn = self.bilinear_gl(first_frames_global, second_frames_local[:, h, w, :])
                        numerator = torch.exp(g_mn.diag())
                        denom = torch.sum(torch.exp(g_mn),1)
                        loss_for_patch = torch.mean(-torch.log(torch.div(numerator, denom)))
                    
                    global_local_loss += loss_for_patch
            global_local_loss /= second_height_range * second_width_range
            
            # local-local loss
            # TODO: this is redundant computation for the encoder
            first_frames_local = self.encoder(first_frames, intermediate_layer=True).permute(0, 2, 3, 1)
            
            # loop through local patches of next frames
            batch_size = frame_batch.shape[0]
            for h in range(second_height_range):
                for w in range(second_width_range):
                    if old_implementation:
                        first_local_times_W = self.linear_ll(first_frames_local[:, h, w, :])
                        f_mn = torch.matmul(first_local_times_W, second_frames_local[:, h, w, :].t())
                        # TODO: this cross entropy loss highly optimized 
                        loss_for_patch = F.cross_entropy(f_mn, torch.arange(batch_size))
                    else:
                        f_mn = self.bilinear_ll(first_frames_local[:, h, w, :], second_frames_local[:, h, w, :])
                        numerator = torch.exp(f_mn.diag())
                        denom = torch.sum(torch.exp(f_mn),1)
                        loss_for_patch = torch.mean(-torch.log(torch.div(numerator, denom)))
                        loss_for_patch = F.cross_entropy(f_mn, torch.arange(batch_size))
                    
                    local_local_loss += loss_for_patch
            local_local_loss /= second_height_range * second_width_range
            
            if batch_index % 5 == 0:
                print(f"batch {batch_index+1}")
                print(f"\tglobal_local_loss: {global_local_loss}")
                print(f"\tlocal_local_loss: {local_local_loss}")
            batch_index += 1
                
            epoch_gl_avg_loss += global_local_loss.detach().item()
            epoch_ll_avg_loss += local_local_loss.detach().item()
            
            if train_mode:
                total_loss = global_local_loss + local_local_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        return epoch_gl_avg_loss / len(frame_batches), epoch_ll_avg_loss / len(frame_batches)