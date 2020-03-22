# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:48:02 2020

@author: Sigve Sørensen
"""

import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

import matplotlib as mlt
import matplotlib.pyplot as plt

import numpy as np
import imageio
import pandas as pd
import pickle


class NeuralNetworkRegression:
    def __init__(self):
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl'")
            
        self.data = self.data[['POWER', 'WS10', 'WS100', 'U10', 'U100']]
        self.data = self.data[['POWER', 'WS100', 'U100']]
        self.columns = self.data.columns
        
    # def net1(self):
    #     """Neural Network"""
        
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda, gpu processing
    #     torch.cuda.device(device) # set device to cuda
    #     torch.manual_seed(1)    # reproducible

    #     feature = 'WS100' # feature variables
    #     target = 'POWER' # target variable

    #     x = torch.tensor(self.data[feature].values.reshape(-1,1)).cuda()  # x data (tensor), shape=(16798, 1)
    #     y = torch.tensor(self.data[target].values.reshape(-1,1)).cuda()  # y data (tensor), shape=(16798, 1)

        
    #     # torch can only train on Variable, so convert them to Variable
    #     x, y = Variable(x), Variable(y)
    #     plt.figure(figsize=(10,4))
    #     plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy(), color = "blue")
    #     plt.title('Regression Analysis')
    #     plt.xlabel('Independent varible')
    #     plt.ylabel('Dependent varible')
    #     plt.savefig('curve_2.png')
    #     plt.show()
        
    #     """Define ANN architecture"""
    #     net = torch.nn.Sequential(
    #             torch.nn.Linear(1, 200),
    #             torch.nn.LeakyReLU(), # Linear activation function
    #             torch.nn.Linear(200, 100),
    #             torch.nn.LeakyReLU(), # Linear activation function
    #             torch.nn.Linear(100, 1),
    #         ).double().cuda()
        
    #     """
    #     optimizer - Define optimization function that updates the weights.
    #                 The first argument in Adam defines which tensors to update.
    #     """
    #     optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    #     loss_func = torch.nn.MSELoss()  # Mean squared error loss
        
    #     BATCH_SIZE = 74 # 16798 / 227 = 74 (all data included)
    #     EPOCH = 227
        
    #     """Deviding into training- and testing sets"""
        
        
    #     torch_dataset = Data.TensorDataset(x, y)
        
    #     loader = Data.DataLoader(
    #         dataset=torch_dataset, 
    #         batch_size=BATCH_SIZE, 
    #         shuffle=True, num_workers=0,)
        
    #     my_images = []
    #     fig, ax = plt.subplots(figsize=(16,10))
    #     i=0
        
    #     # start training
    #     for epoch in range(EPOCH):
    #         i += 1
    #         print(f'Epoch: {i} / {EPOCH}')
    #         for step, (batch_x, batch_y) in enumerate(loader): # for each training step
                
    #             b_x = Variable(batch_x)
    #             b_y = Variable(batch_y)
        
    #             prediction = net(b_x)    # input x and predict based on x
        
    #             loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
        
    #             optimizer.zero_grad()   # clear gradients for next train
    #             loss.backward()         # backpropagation, compute gradients
    #             optimizer.step()        # apply gradients
        
    #             if step == 1:
    #                 # plot and show learning process
    #                 plt.cla()
    #                 ax.set_title('Neural Network Regression Analysis', fontsize=35)
    #                 ax.set_xlabel(f'{feature} (m/s)', fontsize=24)
    #                 ax.set_ylabel(f'{target} (W)', fontsize=24)
    #                 ax.set_xlim(x.cpu().min(), x.cpu().max())
    #                 ax.set_ylim(y.cpu().min(), y.cpu().max())
    #                 ax.hexbin(b_x.cpu().data.numpy(), b_y.cpu().data.numpy(),
    #                           norm=mlt.colors.LogNorm(), cmap='Blues', gridsize=28)
    #                 if i == EPOCH:
    #                     pcm = ax.get_children()[0]
    #                     cb = plt.colorbar(pcm)
    #                     cb.set_label('Data point density', fontsize=13)
    #                 ax.scatter(b_x.cpu().data.numpy(), prediction.cpu().data.numpy(),
    #                            color='k', alpha=0.6, lw=3)
    #                 ax.text(14.0, 0.3, 'Epoch = %d' % epoch,
    #                         fontdict={'size': 24, 'color':  'k'})
    #                 ax.text(14.0, 0.25, 'Loss = %.4f' % loss.cpu().data.numpy(),
    #                         fontdict={'size': 24, 'color':  'k'})
                    
                    
        
    #                 # Used to return the plot as an image array 
    #                 # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
    #                 fig.canvas.draw()       # draw the canvas, cache the renderer
    #                 image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #                 image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
    #                 my_images.append(image)
        
            
        
        
    #     # save images as a gif    
    #     imageio.mimsave('./curve_2_model_3_batch.gif', my_images, fps=12)
        
        
    #     fig, ax = plt.subplots(figsize=(16,10))
    #     plt.cla()
    #     ax.set_title('Neural Network Regression Analysis', fontsize=35)
    #     ax.set_xlabel(f'{feature} (m/s)', fontsize=24)
    #     ax.set_ylabel(f'{target} (W)', fontsize=24)
    #     ax.set_xlim(x.cpu().min(), x.cpu().max())
    #     ax.set_ylim(y.cpu().min(), y.cpu().max())
    #     ax.hexbin(x.cpu().data.numpy(), y.cpu().data.numpy(), cmap='Blues',
    #               norm=mlt.colors.LogNorm(), gridsize = 28)
    #     pcm = ax.get_children()[0]
    #     cb = plt.colorbar(pcm, ax=ax,)
    #     cb.set_label('Data point density', fontsize=13)
    #     prediction = net(x)     # input x and predict based on x
    #     ax.scatter(x.cpu().numpy(), prediction.cpu().data.numpy(), color='k',
    #                alpha=0.6, lw=0.3)
    #     plt.savefig('curve_2_model_3_batches.png')
    #     plt.show()
        
        
        
        
        
        
        
        
    def net2(self):
        """Neural Network"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda, gpu processing
        torch.cuda.device(device) # set device to cuda
        torch.manual_seed(1)    # reproducible

        feature = 'WS100' # feature variables
        target = 'POWER' # target variable

        X = torch.tensor(self.data[feature].values.reshape(-1,1)).cuda()  # x data (tensor), shape=(16798, 1)
        y = torch.tensor(self.data[target].values.reshape(-1,1)).cuda()  # y data (tensor), shape=(16798, 1)
        
        """Deviding into training- and testing sets"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # torch can only train on Variable, so convert them to Variable
        X_train, y_train = Variable(X_train), Variable(y_train)
        
        # plt.figure(figsize=(10,4))
        # plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy(), color = "blue")
        # plt.title('Regression Analysis')
        # plt.xlabel('Independent varible')
        # plt.ylabel('Dependent varible')
        # plt.savefig('curve_2.png')
        # plt.show()
        
        """Define ANN architecture"""
        net = torch.nn.Sequential(
                torch.nn.Linear(1, 200),
                torch.nn.LeakyReLU(), # Linear activation function
                torch.nn.Linear(200, 100),
                torch.nn.LeakyReLU(), # Linear activation function
                torch.nn.Linear(100, 1),
            ).double().cuda()
        
        """
        optimizer - Define optimization function that updates the weights.
                    The first argument in Adam defines which tensors to update.
        """
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_func = torch.nn.MSELoss()  # Mean squared error loss
        
        BATCH_SIZE = 74 # 16798 / 227 = 74 (all data included)
        EPOCH = 227
        
        torch_dataset = Data.TensorDataset(X_train, y_train)
        
        loader = Data.DataLoader(
            dataset=torch_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=0,)
        
        my_images = []
        fig, ax = plt.subplots(figsize=(16,10))
        i=0
        losses = []
        preds = []
        
        """Train model"""
        for epoch in range(EPOCH):
            i += 1
            print(f'Epoch: {i} / {EPOCH}')
            for step, (batch_x, batch_y) in enumerate(loader): # for each training step
                
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
        
                prediction = net(b_x)    # input x and predict based on x
                preds.append(prediction)
        
                loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
                losses.append(loss)
        
                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
        
                if step == 1:
                    # plot and show learning process
                    plt.cla()
                    ax.set_title('Neural Network Regression Analysis', fontsize=35)
                    ax.set_xlabel(f'{feature} (m/s)', fontsize=24)
                    ax.set_ylabel(f'{target} (W)', fontsize=24)
                    ax.set_xlim(X.cpu().min(), X.cpu().max())
                    ax.set_ylim(y.cpu().min(), y.cpu().max())
                    ax.hexbin(b_x.cpu().data.numpy(), b_y.cpu().data.numpy(),
                              norm=mlt.colors.LogNorm(), cmap='Blues', gridsize=28)
                    if i == EPOCH:
                        pcm = ax.get_children()[0]
                        cb = plt.colorbar(pcm)
                        cb.set_label('Data point density', fontsize=13)
                    ax.scatter(b_x.cpu().data.numpy(), prediction.cpu().data.numpy(),
                               color='k', alpha=0.6, lw=3)
                    ax.text(14.0, 0.3, 'Epoch = %d' % epoch,
                            fontdict={'size': 24, 'color':  'k'})
                    ax.text(14.0, 0.25, 'Loss = %.4f' % loss.cpu().data.numpy(),
                            fontdict={'size': 24, 'color':  'k'})
                    
                    
        
                    # Used to return the plot as an image array 
                    # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
                    fig.canvas.draw()       # draw the canvas, cache the renderer
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
                    my_images.append(image)
        
            
        
        
        # save images as a gif    
        imageio.mimsave('./curve_2_model_3_batch.gif', my_images, fps=12)
        
        
        fig, ax = plt.subplots(figsize=(16,10))
        plt.cla()
        ax.set_title('Neural Network Regression Analysis', fontsize=35)
        ax.set_xlabel(f'{feature} (m/s)', fontsize=24)
        ax.set_ylabel(f'{target} (W)', fontsize=24)
        ax.set_xlim(X.cpu().min(), X.cpu().max())
        ax.set_ylim(y.cpu().min(), y.cpu().max())
        ax.hexbin(X.cpu().data.numpy(), y.cpu().data.numpy(), cmap='Blues',
                  norm=mlt.colors.LogNorm(), gridsize = 28)
        pcm = ax.get_children()[0]
        cb = plt.colorbar(pcm, ax=ax,)
        cb.set_label('Data point density', fontsize=13)
        prediction = net(X_test)     # input x and predict based on x
        loss = loss_func(prediction, y_test)
        ax.scatter(X_test.cpu().numpy(), prediction.cpu().data.numpy(), color='k',
                   alpha=0.6, lw=0.3)
        ax.text(14.0, 0.25, 'Loss = %.4f' % loss.cpu().data.numpy(),
                            fontdict={'size': 24, 'color':  'k'})
        plt.savefig('curve_2_model_3_batches.png')
        plt.show()

        
        
if __name__ == '__main__':
    tmp = NeuralNetworkRegression()
    tmp.net2()
                
                
