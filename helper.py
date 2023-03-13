from enum import Enum
from collections import namedtuple


#Colors
WHITE = (255,255,255)
RED = (255,0,0)
BLACK = (0,0,0)
BLUE = (0,100,255)
BLUE_BORDER = (0,0,255)
HEAD_COLOR = (100,80,235)
HEAD_BORDER = (102,0,204)


BLOCK_SIZE = 20
SPEED = 100 #snake speed
SEARCH_TIME = 100 #more value => more time for snake to search food. Multiplied with len(body); exceeding which game-over
BATCH_SIZE = 1000

#Everywhere direction is taken in order: 
# Absolute: UP-RIGHT-DOWN-LEFT with values 0-1-2-3
# Relative: LEFT-FRONT-RIGHT
# Some part of code is written with this assumption
#Directions
class Direction(Enum):
    #do not change the values; they're used to get the relative directions
    UP=0
    RIGHT=1
    DOWN=2
    LEFT=3

#Events
class States(Enum):
    PLAYING = 0
    GAME_OVER = -10
    ATE_FOOD = 10


Point = namedtuple('Point',['x','y'])


#Plot the scores
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
matplotlib.rcParams['toolbar'] = 'None' #hides Home, Zoom, etc buttons

plt.ion() #interactive on; plots interactively
def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf()) #get current figure; create new if none exists
    plt.clf() #clear current figure
    plt.gcf().canvas.manager.set_window_title("Training...")
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False) #show the plot but dont block the program
    plt.pause(0.1) #update the display before this; pause to perform event tasks
