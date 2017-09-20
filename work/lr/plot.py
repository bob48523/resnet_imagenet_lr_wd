#!/usr/bin/env python3

import argparse
import os
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

def main():
    trainP = [
         os.path.join('lr0.5/', 'train.csv'),
         os.path.join('lr0.3/', 'train.csv'),
         os.path.join('lr0.1/', 'train.csv'),
         os.path.join('lr0.03/', 'train.csv'),
         os.path.join('lr0.01/', 'train.csv')
    ]
    testP = [
         os.path.join('lr0.5/', 'test.csv'),
         os.path.join('lr0.3/', 'test.csv'),
         os.path.join('lr0.1/', 'test.csv'),
         os.path.join('lr0.03/', 'test.csv'),
         os.path.join('lr0.01/', 'test.csv')
    ]
    lr = ['0.5','0.3','0.1','0.03','0.01']
    N = 625 # Rolling loss over the past epoch.
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    for i in range(5):
        trainData = np.loadtxt(trainP[i], delimiter=',').reshape(-1, 3)
        testData = np.loadtxt(testP[i], delimiter=',').reshape(-1, 3)
        trainI, trainLoss, trainErr = np.split(trainData, [1,2], axis=1)
        trainI, trainLoss, trainErr = [x.ravel() for x in
                                         (trainI, trainLoss, trainErr)] 
        trainI_, trainLoss_, trainErr_ = rolling(N, trainI, trainLoss, trainErr)
        testI, testLoss, testErr = np.split(testData, [1,2], axis=1)
        #plt.plot(trainI_, trainLoss_, label='Train'+lr[i])
        plt.plot(testI, testLoss, label='Test'+lr[i])
        
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    ax1.set_yscale('log')   
    loss_fname = os.path.join('loss.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    for i in range(5):
        trainData = np.loadtxt(trainP[i], delimiter=',').reshape(-1, 3)
        testData = np.loadtxt(testP[i], delimiter=',').reshape(-1, 3)
        trainI, trainLoss, trainErr = np.split(trainData, [1,2], axis=1)
        trainI, trainLoss, trainErr = [x.ravel() for x in
                                         (trainI, trainLoss, trainErr)] 
        #trainI_, trainLoss_, trainErr_ = rolling(N, trainI, trainLoss, trainErr)
        testI, testLoss, testErr = np.split(testData, [1,2], axis=1)
 
        #plt.plot(trainI_, trainErr_, label='Train'+lr[i])
        plt.plot(testI, testErr, label='Test'+lr[i])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    ax2.set_yscale('log')
    plt.legend()
    err_fname = os.path.join('error.png')
    plt.savefig(err_fname)
    print('Created {}'.format(err_fname))
    loss_err_fname = os.path.join('loss-error.png')
    os.system('convert +append {} {} {}'.format(loss_fname, err_fname, loss_err_fname))
    print('Created {}'.format(loss_err_fname))

def rolling(N, i, loss, err):
    i_ = i[N-1:]
    K = np.full(N, 1./N)
    loss_ = np.convolve(loss, K, 'valid')
    err_ = np.convolve(err, K, 'valid')
    return i_, loss_, err_

if __name__ == '__main__':
    main()
