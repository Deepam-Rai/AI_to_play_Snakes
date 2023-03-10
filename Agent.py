# Main controller component
# It handles the Snake game and the AI Mind and also QTrainer.
# 1. Gets state from SnakeGame
# 2. Gives to Mind and gets next move
# 3. Gives to game and gets next_state, reward for this action, gameover or not
# 4. Gives state, action, reward, next_state, game_over or not to QTrainer for training model

# Memory Relay Concept
# For every action - if it is non-terminating - model is trained
#   - called train-short-term-memory
# Further we also memorize that state, reward, action, next_state, game_over in long-term-memory
# When game is over
#   All the steps taken in that game - which is memorized
#   is again fed to model for train - called train-long-term-memory
# Long term memory is not cleared at the end of the game - instead when limit is reached, pops off last item automatically(deque)

# Theory
# We use RL Q-learning to train the Model.
# Mind is a neural network Model 
#       that gives us the predicted Q-value for the current state.
#    This is the 'predicted' value of model.
# To train it  -  
#       we get an action from Mind
#       perform that action,
#       get new states,
#       predict Q-values for this new states
#       update the Q-value of previous state with this new state Q-value using Bellman-Equation
#   This updated Q-value becomes our 'target' value.
# We find the Loss(predicted value, target value) and update the parameters



from collections import deque
import numpy as np
import random

from Snake_Game import SnakeGame
from Mind import Mind, QTrainer

from helper import *

MAX_MEMORY = 100_000

class Agent:
    def __init__(self, model=None, trainer=None, epsilon=None):
        ''' The orchestrator.
        model(Mind) and trainer(QTrainer) - if not given creates its own.
        epsilon: the random factor by which random action is chosen. If not provided hardcoded formula used to determine.
        Game is created during training. Reinitialized and used on game over till training continues.
        '''
        self.model = Mind(19,[512],3) if model is None else model
        self.trainer = QTrainer(self.model) if trainer is None else trainer
        self.memory = deque(maxlen=MAX_MEMORY)
        self.n_games = 0
        self.epsilon = epsilon
        self.gameWidth = 680
        self.gameHeight = 480
        #game is initialized in the self.train function
    
    def get_state(self, game):
        ''' Returns nympy.ndarray with values in sequence:
        '''
        state = []
        head = game.snake.head
        #surrounding points
        s_points = [
            Point(head.x, head.y - BLOCK_SIZE), #up
            Point(head.x + BLOCK_SIZE, head.y), #right
            Point(head.x, head.y + BLOCK_SIZE), #down
            Point(head.x - BLOCK_SIZE, head.y) #Left
        ]
        
        #snake direction in absolute direction
        abs_dirs = [
            game.snake.direction == Direction.UP,
            game.snake.direction == Direction.RIGHT,
            game.snake.direction == Direction.DOWN,
            game.snake.direction == Direction.LEFT
        ]

        rel_dirs = [
            game.snake.rel_to_abs_dir('LEFT'),
            game.snake.rel_to_abs_dir('FRONT'),
            game.snake.rel_to_abs_dir('RIGHT'),
        ]
        dangers = [
            game.collision( s_points[ rel_dir.value]) for rel_dir in rel_dirs
        ]
        food_rel = game.snake.rel_dir_from_head(game.food)
        #body awareness
        #   give the relative direction of tail and mid from head.
        blocks = [game.snake.body[-1], game.snake.body[ len(game.snake.body)//2-1]]
        body_rel_dirs = []
        for x in blocks:
            body_rel_dirs.extend(game.snake.rel_dir_from_head(x))
        
        state.extend(dangers)
        state.extend(abs_dirs)
        state.extend(food_rel)
        state.extend(body_rel_dirs)
        state = np.array(state, dtype=int)
        return state

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append( (state, action, reward, next_state, game_over) )
        #automatically pops oldest item when limit reached
    
    def train_long_term_memory(self):
        '''Memory Relay concept implementation
        Takes the data from memory.
        Makes them into batches.
        Passes to trainer for training.
        # sampling while creating batches to reduce the sequence-correlation'''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = random.sample(self.memory, len(self.memory))
        # from [(1,2,3),(1,2,3),(1,2,3)...] to [1,1,1,..], [2,2,..],[3,3...]
        states, actions, rewards, next_states, game_overs = zip(*mini_sample) #unzip operation
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
    
    def train_short_term_memory(self,state_old, move, reward, state_new, game_over):
        '''Takes the state, action,...
           passes to trainer for training. Trainer makes it into batch(with one item) and trains model.'''
        self.trainer.train_step(state_old, move, reward, state_new, game_over)

    def train(self, gameWidth=None, gameHeight=None):
        '''Trains self.model using self.trainer by creating a game(which is reinitialized and used).
        Inputs gameWidth and gameHeight - window size for games - if not given default value is used.
        Also plots the scores
        '''
        plot_scores = []
        plot_mean_scores = [] #TODO make moving average
        total_score = 0
        highest_score = 0
        gameWidth = self.gameWidth if gameWidth is None else gameWidth
        gameHeight = self.gameHeight if gameHeight is None else gameHeight
        
        game = SnakeGame(width=gameWidth, height=gameHeight)
        while True:
            #get old state
            state_old = self.get_state(game)
            #get move
            epsilon = (150-self.n_games)/350 if self.epsilon is None else self.epsilon
            # epsilon = 0.01 if epsilon<0 else epsilon
            move = self.model.get_move(state_old, epsilon)

            #perform move and get new state
            reward, game_over, score = game.move(move)
            state_new = self.get_state(game)
            #train short term memory
            self.train_short_term_memory(state_old, move, reward, state_new, game_over)
            
            #remember
            self.remember(state_old, move, reward, state_new, game_over)

            #if game over then Memory Replay
            if game_over:
                game.reset()
                self.n_games +=1
                self.train_long_term_memory()

                if score>highest_score:
                    highest_score=score
                    self.model.save()
                print(f'Game:{self.n_games}\t Score:{score}\t Highest:{highest_score}')

                #plot
                plot_scores.append(score)
                total_score += score
                mean_score = total_score/self.n_games #TODO make moving average
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

import torch

if __name__=='__main__':
    model = None
    epsilon=None
    
    #-------
    #load pre-trained model; comment below to not train on pretrained model
    model = Mind(19,[512],3)
    model.load_state_dict(state_dict=torch.load('./model/trained_model.pth'))
    epsilon=0
    #TODO: Load the trainer too; since Adam optimizer is being used; Nevertheless its working fine :)
    #-------
    
    
    agent = Agent(model=model, epsilon=epsilon)
    agent.train()
