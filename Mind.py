# The mind of the AI.
#  Theoretically:
#  The mind is an neural-network that approximates the Q-values of given state: Q(s,a)
# Q-values means the quality of action.
# Q(s,a) = Max cumulative reward that i can gain from this state with further subsequent states also being optimum(getting max reward)
# Thus the output of the Mind:
#  an array whose each element corresponds to a possible action that can be taken.
#  these values can be interpreted as the Q(current state, that action taken) - Q-value if we take that action.
# Since we want max reward - we take the action with highest Q-value - thus action with higher value is taken.

# QTrainer takes the data from Agent and trains the mind using Q-Learning

import torch
import torch.nn as nn
import torch.optim as optim

import os #to save the model

import numpy as np
from numpy import random

class Mind(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, epsilon=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.epsilon = epsilon #random moves; exploration; [0,1]

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dims[0], self.output_dim),
        )
    
    def forward(self,x):
        h = self.network(x)
        return h
    
    def get_move(self, state, epsilon=None):
        '''state: game state
        epsilon: exploration vs exploitation; range [0,1]. If a random(0,1)<epsilon then take random move, else think and move.
          if None then default epsilon taken.'''
        epsilon = self.epsilon if epsilon is None else epsilon
        move = [0,0,0]
        if random.rand()<epsilon:
            #random move
            next_dir = random.randint(0,3)
            move[next_dir]=1
        else:
            stateTen = torch.tensor(state, dtype=torch.float)
            prediction = self(stateTen)
            next_dir = torch.argmax(prediction).item()
            move[next_dir]=1
        return move

    def save(self, filename='model.pth'):
        #TODO: save the optimizer dict too.
        #ensure the path is valid
        folder_path = './model'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pathName = os.path.join(folder_path, filename)
        #save the model
        torch.save( self.state_dict(), pathName)


class QTrainer:
    def __init__(self, model, gamma=0.9, lr=0.001):
        '''model: Mind()
           gamma: discount factor for future moves; using Q-learning; Bellman Equation;
           lr: for the Adam optimizer
           '''
        self.model = model
        self.gamma = gamma
        self.lr = lr

        #TODO: save teh optimizer dict too
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, states, actions, rewards, next_states, game_overs):
        '''
        states: the old state
        actions: action taken in old state
        rewards: reward gained from action taken in old state
        next_states: new state from old state after the action was taken
        game_overs: whether new state is terminating or not; whether the action taken terminated the game or not.

        Accepts a tuple entries too: If the inputs are not in batches, then the function converts them to batches and work.
        '''
        # 1) PREPARE THE DATA
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)

        #they should be in batches; if not make them
        if len(states.shape)==1: #if 1D
            states = torch.unsqueeze(states,0)
            actions = torch.unsqueeze(actions,0)
            rewards = torch.unsqueeze(rewards, 0)
            next_states = torch.unsqueeze(next_states, 0)
            game_overs = (game_overs,)
        
        # 2) GET PREDICTIONS
        #predicted Q-values of actions of current state
        predQ = self.model(states)

        # 3) GET TARGET VALUES
        target = predQ.clone()
        for i in range(len(game_overs)):
            Q_new = rewards[i] #current state reward
            if not game_overs[i]: #if the action was non-terminating
                #apply bellman equation
                Q_new = rewards[i] + self.gamma * torch.max(self.model(next_states[i]))
            target[i][torch.argmax(actions[i]).item()] = Q_new
        # 4) CLEAR OLD GRADIENTS
        self.optimizer.zero_grad()
        # 5) CALCULATE LOSS
        loss = self.criterion(predQ, target)
        # 6) CALCULATE GRADIENTS
        loss.backward()
        # 7) UPDATE PARAMETERS
        self.optimizer.step()


if __name__=='__main__':
    M = Mind(5,[256,256],3)
    print(M.test())