# Snake game
# Class SnakeGame:
#   Serves as environment for Snake body(Not Mind)
#   has snake inside it only; we can give next direction for moving via SnakeGame
#       Direction is given as (bool, bool, bool) for (left, front, right)


import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1' #to hide the welcome greetings
import pygame
import random

from helper import *


# Snake class
class Snake:
    def __init__(self, init_direction, head):
        self.direction = init_direction #initial direction
        if not isinstance(head, Point):
            raise TypeError('Initial Head position must of of type Point')
        else:
            self.head=head
            self.body = [self.head,   #start with head +2 extra blocks
                    Point(self.head.x - BLOCK_SIZE, self.head.y),
                    Point(self.head.x-2*BLOCK_SIZE, self.head.y) ]

    def collision(self, point=None):
        '''Checks if given point collides with the snake body
        If no point is provided, then check if its head collides with its body.'''
        check_from = 0 #check from where in body
        if point is None:
            point=self.head
            check_from = 1 #dont check the head
        if point in self.body[check_from:]:
            return True #collision ocurred
        return False #no collision
        #TODO: when snake moves; new head is added but tail is not removed; in that instant snake is 1 block longer

    def play_step(self, direction):
        '''direction: (left, front, right) in bool;
        Moves the snake in given direction.
        Inserts head. [DOES NOT DELETE THE LINGERING TAIL] [game deletes it after checking ate_food]
        Returns: GAME_OVER if collision with snake body else PLAYING.'''
        self._move(direction) #just updates head
        self.body.insert(0,self.head) #insert head block
        if self.collision(): #collided on itself
            return States.GAME_OVER
        return States.PLAYING

    def _move(self, direction):
        '''direction: tuple of (left, front, right) in bool of movement.
        Helper function for play_step(). It just updates the head doesnt add block.
        '''
        left, front, right = direction
        new_dir = self.direction #default go front
        if left:
            new_dir = self.rel_to_abs_dir('LEFT')
        if right:
            new_dir = self.rel_to_abs_dir('RIGHT')
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        self.head = Point(x,y) #the new head position
    
    def rel_to_abs_dir(self, rel_dir):
        ''' Relative Direction -> Absolute Direction
         if current is Direction.DOWN and ask for relative LEFT then return Direction.RIGHT'''
        c_dir = self.direction
        if rel_dir=='RIGHT':
            return Direction( (c_dir.value+1)%4 )
        elif rel_dir=='LEFT':
            return Direction( (c_dir.value-1)%4 )
        elif rel_dir=='FRONT':
            return Direction( (c_dir.value) )
        elif rel_dir=='BACK':
            return Direction( (c_dir.value+2)%4 )
    def abs_to_rel_dir(self, abs_dir):
        '''Absolute -> Relative Direction
        If current dir= UP; given abs_dir=DOWN; returns BACK'''
        rels = ['FRONT','RIGHT','BACK','LEFT']
        for rel in rels:
            if self.rel_to_abs_dir(rel)==abs_dir:
                return rel
    def abs_dir_from_head(self, point):
        '''Returns [None or Direction.direction,..] array for whichever directions hold else None.
        # Tells in Absolute Directions where the point lies from the head.'''
        result = [
            Direction.UP if point.y<self.head.y else None,
            Direction.RIGHT if point.x>self.head.x else None,
            Direction.DOWN if point.y>self.head.y else None,
            Direction.LEFT if point.x<self.head.x else None
        ]
        return result
    def rel_dir_from_head(self, point):
        ''' Returns ['FRONT','RIGHT','BACK','LEFT'] as True if Holds else False
        # tells relative direction where point lies from the head.'''
        abs_dir = self.abs_dir_from_head(point)
        rel_dir = [False, False, False, False]
        notations = ['FRONT','RIGHT','BACK','LEFT']
        for a_dir in abs_dir:
            if a_dir is not None:
                rel_dir[notations.index( self.abs_to_rel_dir(a_dir))] = True
        return rel_dir

    def delete_last(self):
        '''delete the last tail block'''
        self.body.pop()
    
    def corners(self):
        ''' return the 4 extreme parts of the body.
        #TODO: Check if this function works properly.  havent used this function yet :)'''
        top_l = self.head
        top_r = self.head
        right_t = self.head
        right_b = self.head
        bot_r = self.head
        bot_l = self.head
        left_b = self.head
        left_t = self.head
        for p in self.body[1:]:
            if (p.x>right_t.x) or (p.x==right_t.x and p.y<right_t.y):
                right_t = p
            if (p.x>right_b.x) or (p.x==right_b.x and p.y>right_b.y):
                right_b = p
            if (p.y>bot_r.y) or (p.y==bot_r.y and p.x>bot_r.x):
                bot_r = p
            if (p.y>bot_l.y) or (p.y==bot_l.y and p.x<bot_l.x):
                bot_l = p
            if (p.x<left_b.x) or (p.x==left_b.x and p.y>left_b.y):
                left_b = p
            if (p.x<left_t.x) or (p.x==left_b.x and p.y<left_t.y):
                left_t = p
            if (p.y<top_l.y) or (p.y==top_l.y and p.x<top_l.x):
                top_l = p
            if (p.y<top_r.y) or (p.y==top_r.y and p.x>top_r.x):
                top_r = p
        rels = [top_l, top_r, right_t, right_b, bot_r, bot_l, left_b, left_t]
        for i in range(len(rels)):
            rels[i] = Point(rels[i].x - self.head.x, rels[i].y - self.head.y)
        return rels

    def draw(self, display):
        for b in self.body:
            pygame.draw.rect(display, BLUE_BORDER, pygame.Rect(b.x, b.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(display, BLUE, pygame.Rect(b.x+4, b.y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))
            

# Game class
class SnakeGame:
    def __init__(self,height, width):
        '''height: window height
            width: window width
        '''
        self.height = height #window height and width
        self.width = width
        #init pygame display
        pygame.init()
        self.font = pygame.font.Font('arial.ttf', 25)
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock() #create clock to track time
        self.reset()
    
    def reset(self):
        '''Initialize the game state'''
        #the snake
        self.snake = Snake(Direction.RIGHT, Point(self.width/2, self.height/2))
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration=0
        self._since_last_ate = 0 #snake should eat asap
    
    def _place_food(self):
        # generate random coordinate
        x = random.randint(0, (self.width-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.height-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        #subtraction because starting from 0; int division gives us max block index; mult gives us the coordinate
        self.food = Point(x,y)
        if self.snake.collision(self.food):
            self._place_food() #recursive call until free cell is reached
        #TODO:if no free cell it will forever search for free cell.
    
    def move(self, dir):
        '''dir: (bool, bool, bool) for (LEFT, FRONT, RIGHT) direction. One should be True.
        Moves the snake.
        Checks collisions.
        If the snake is roaming without eating for long time, then also returns GAME_OVER=True. Using frame_iterations. Control using SEARCH_TIME in helper.py
        Returns: <int>reward, <bool> game_over, <int>score '''
        self._since_last_ate +=1
        reward = 0
        game_over = True if self.snake.play_step(dir)==States.GAME_OVER else False #collision with snake body checked here
        if not game_over:
            if not self.collision_boundary(self.snake.head): #collision
                if self.snake.head==self.food: #ate food
                    self.score +=1
                    self._since_last_ate = 0
                    self._place_food()
                    reward = States.ATE_FOOD.value
                else:
                    self.snake.delete_last() #snake moved
            else:
                game_over = True
                reward = States.GAME_OVER.value
        else:
            reward = States.GAME_OVER.value
        #for roaming/looping snake.
        if self._since_last_ate > SEARCH_TIME*len(self.snake.body):
            game_over = True
            reward = States.GAME_OVER.value
        self.update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score
    
    def collision(self, point):
        '''Checks if point collides with (i) boundaries or (ii) snake.
        Returns bool.'''
        return self.collision_boundary(point) or self.snake.collision(point)

    def collision_boundary(self, point):
        if point.x > self.width-BLOCK_SIZE or point.x<0 or point.y>self.height-BLOCK_SIZE or point.y<0:
            return True
        return False

    def update_ui(self):
        #clear the screen
        self.display.fill(BLACK)
        #snake
        self.snake.draw(self.display)
        #food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        #scores
        text = self.font.render(f'Score:{self.score}',True, WHITE )
        self.display.blit(text, [0,0])
        pygame.display.flip() #updates the window


if __name__=='__main__':
    # USER PLAYS
    #init the game
    pygame.init()

    game = SnakeGame(width=640, height=480)

    game_over = False    
    #Game Loop
    while not game_over:
        game.frame_iteration +=1

        #get user input
        dir = (False, True, False) #default move front
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    dir = (True, False, False)
                elif event.key == pygame.K_UP:
                    dir = (False, True, False)
                elif event.key == pygame.K_RIGHT:
                    dir = (False, False, True)
        
        #Move the snake
        reward, game_over, score  = game.move(dir)
        
        #update UI
        game.update_ui()
        #framerate
        game.clock.tick(SPEED) 
    print("Game Over!!")
    print(f'Score:{game.score}')

