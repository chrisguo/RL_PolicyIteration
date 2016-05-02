
import itertools
import numpy as np
from Game import CardGame
import random
from Logging import func_timer
import pickle
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pdb
from collections import defaultdict


"""
    Module for import statements required by solution and general dictionary initialisations and plotting functions

"""

def dict_init(state_dim = (10,21), actions = (0,1), init_ps = 0):

     # list of states
    all_states= list(itertools.product(range(1,state_dim[0]+1), range(1,state_dim[1]+1)))

    # list of actions
    all_actions = []
    for i in range(1,11): #number of dealer first cards
        for j in range(1,22): #number of potential player sums

            all_actions.append(0) if j>=init_ps else all_actions.append(1)

    # creation of dictionary
    policy = dict(zip(all_states, all_actions))
    policy[(0,0)] = 0

    # Create action value function
    # Mapping keys;state & action to value - initialise to zero
    num_actionVals = 10*21*2
    all_values = np.zeros(num_actionVals)
    all_state_actions = list(itertools.product(all_states, actions))
    Q_sa= dict(zip(all_state_actions, all_values ))
    Q_sa[((0,0),0)] = 0
    Q_sa[((0,0),1)] = 0

    return Q_sa, policy

def plotValueFcn(V):

    # 3 d mesh plot for value function
    X, Y  = np.meshgrid(np.arange(1,11),np.arange(1,21))
    Z = np.zeros(X.shape)
    for i in range(1,11):
        for j in range(1,21):
            Z[j-1,i-1] = V[(i,j)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection ='3d')
    ax.plot_wireframe(X, Y, Z, rstride =1, cstride = 1)
    ax.set_xlabel('Dealer card')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('Value')
    plt.savefig('output/plots/MC_Val.png')
    plt.show()



def plotMSE(num_its,lam, mse_lam, mse_lam_learn, name):

    # MSE Vs lambda and MSE learning curve plots

    plt.plot(lam, mse_lam)
    plt.xlabel('Lambda')
    plt.ylabel('Mean Squared Error')
    plt.savefig('output/plots/'+name+'_LamVsMSE.png')
    plt.show(block = False)
    plt.close()

    episode = range(0,num_its, (num_its/100))
    plt.plot(episode, mse_lam_learn[0], 'r-', label = "Lambda = 0")
    plt.plot(episode, mse_lam_learn[10], 'b-', label = "Lambda = 1")
    plt.xlabel('Episode')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig('output/plots/'+name+'_LamVsEpisode.png')
    plt.show(block = False)