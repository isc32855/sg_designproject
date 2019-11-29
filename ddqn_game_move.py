# In[ ]:


# -*- coding: utf-8 -*-
from __future__ import division, print_function
import collections
import numpy as np
import pygame
import random
import os
import math
DATA_DIR = "C:/Users/82103/Desktop/DesignProject/jupyter/bf"

class MyBeamforming_game_1(object):

    def __init__(self):
        
        pygame.init()
        pygame.key.set_repeat(10, 100)
        # set constants
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_GREEN = (64, 255, 64)
        self.COLOR_BLACK = (0, 0, 0)
        self.GAME_WIDTH = 400
        self.GAME_HEIGHT = 400
        self.GAME_FLOOR = 300
        self.USER_WIDTH = 30
        self.USER_HEIGHT = 30
        self.USER_RADIUS = 200
        self.BEAM_WIDTH = 150
        self.BEAM_HEIGHT = 30
        self.CENTER_x = 200
        self.CENTER_y = 200
        self.USER_VELOCITY = 5                                                 
        self.BEAM_VELOCITY = 5                     
        self.FONT_SIZE = 30
        self.CUSTOM_EVENT = pygame.USEREVENT + 1
        self.font = pygame.font.SysFont("Comic Sans MS", self.FONT_SIZE)
        self.MAX_FRAME = 32
        
    def reset(self):
        self.frames = collections.deque(maxlen=4)                          
        self.game_over = False
        self.game_score = 0
        self.reward = 0
        self.diff_av = 0
        # initialize positions
        self.beam_a = 90
        self.beam_r = math.radians(self.beam_a)                      
        self.beam_x = -0.5*self.BEAM_HEIGHT*math.sin(self.beam_r)+self.CENTER_x
        self.beam_y = -0.5*self.BEAM_HEIGHT*math.cos(self.beam_r)+self.CENTER_y
        self.user_a = random.randrange(30, 150, self.BEAM_VELOCITY)
        self.user_r = math.radians(self.user_a)  
        self.user_x = round(self.USER_RADIUS*math.cos(self.user_r))+self.CENTER_x
        self.user_y = -round(self.USER_RADIUS*math.sin(self.user_r))+self.CENTER_y
        # set up display, clock, etc
        self.screen = pygame.display.set_mode((self.GAME_WIDTH, self.GAME_HEIGHT))
        pygame.display.set_caption('DTL-Design Project')
        
    def step(self, action, num_frame, reward):
        pygame.event.pump()        
        self.first_diff = abs(self.user_a-self.beam_a)
        self.reward = reward
        self.num_frame = num_frame
        self.num_frame += 1

        ### beam
        if action == 1:   # move beam left
            if self.beam_a >= 150:
                self.beam_a = 150
            else:
                 self.beam_a += self.BEAM_VELOCITY
            self.beam_r = math.radians(self.beam_a)
            self.beam_x = -0.5*self.BEAM_HEIGHT*math.sin(self.beam_r)+self.CENTER_x
            self.beam_y = -0.5*self.BEAM_HEIGHT*math.cos(self.beam_r)+self.CENTER_y
        elif action == 2: # move beam right
            if self.beam_a <= 30:
                self.beam_a = 30
            else:
                self.beam_a -= self.BEAM_VELOCITY
            self.beam_r = math.radians(self.beam_a)
            self.beam_x = -0.5*self.BEAM_HEIGHT*math.sin(self.beam_r)+self.CENTER_x
            self.beam_y = -0.5*self.BEAM_HEIGHT*math.cos(self.beam_r)+self.CENTER_y

        ### user
        if self.num_frame % 4 == 0 and self.num_frame < 32: 
            self.user_action = random.randint(0, 9)
        else:
            self.user_action = 2
    
        if self.user_action == 0:   ### move user left
            if self.user_a >= 150:
                self.user_a = 150
            else:
                self.user_a += self.USER_VELOCITY
            self.user_r = math.radians(self.user_a)
            self.user_x = round(self.USER_RADIUS*math.cos(self.user_r))+self.CENTER_x
            self.user_y = -round(self.USER_RADIUS*math.sin(self.user_r))+self.CENTER_y
        elif self.user_action == 1:  ### move user right
            if self.user_a <= 30:
                self.user_a = 30
            else:
                self.user_a -= self.USER_VELOCITY
            self.user_r = math.radians(self.user_a)
            self.user_x = round(self.USER_RADIUS*math.cos(self.user_r))+self.CENTER_x
            self.user_y = -round(self.USER_RADIUS*math.sin(self.user_r))+self.CENTER_y  
        else:
            pass
        
        self.screen.fill(self.COLOR_GREEN, (0, 0, self.GAME_WIDTH, self.GAME_HEIGHT))
    
        city = pygame.image.load(os.path.join(DATA_DIR,'crossroad.png'))
        city = pygame.transform.scale(city, (self.GAME_WIDTH, self.GAME_HEIGHT))
        self.screen.blit(city, (0, 0))

        # draw City
        building = pygame.image.load(os.path.join(DATA_DIR,'building.png'))
        building = pygame.transform.scale(building, (self.GAME_WIDTH//8, self.GAME_HEIGHT//4))
        self.screen.blit(building, (self.CENTER_x - self.GAME_WIDTH//16, self.GAME_HEIGHT//2+10))
        
        # update user position
        pygame.draw.rect(self.screen, self.COLOR_WHITE,[self.user_x, self.user_y,self.USER_WIDTH,self.USER_HEIGHT])
        self.user = pygame.image.load(os.path.join(DATA_DIR,'smartphone.png'))
        self.user = pygame.transform.scale(self.user, (self.USER_WIDTH,self.USER_HEIGHT))
        self.screen.blit(self.user, (self.user_x, self.user_y))
    
        # update beam position
        original_beam = pygame.image.load(os.path.join(DATA_DIR,'ellipse.png'))
        original_beam = pygame.transform.scale(original_beam, (self.BEAM_WIDTH,self.BEAM_HEIGHT))
        self.beam = pygame.transform.rotate(original_beam, self.beam_a)
        self.rect = self.beam.get_rect()
        self.rect.center = (self.CENTER_x + (self.BEAM_WIDTH/2)*math.cos(self.beam_r), 
                            self.CENTER_y - (self.BEAM_WIDTH/2)*math.sin(self.beam_r))
        self.screen.blit(self.beam, self.rect)
       
        # draw Antenna 
        pygame.draw.polygon(self.screen, self.COLOR_WHITE, [[self.CENTER_x + 10,self.CENTER_y], 
                                                            [self.CENTER_x - 10,self.CENTER_y], 
                                                            [self.CENTER_x, self.CENTER_y + 10]],1)
        pygame.draw.lines(self.screen, self.COLOR_WHITE, False, [[self.CENTER_x,self.CENTER_y], 
                                                                 [self.CENTER_x,self.CENTER_y + 25], 
                                                                 [self.CENTER_x + 10,self.CENTER_y + 25]],1)
                                        
        ### for visualization
        # store the difference between user_a and beam_a
#        self.diff_av += abs(self.user_a-self.beam_a)/self.MAX_FRAME
        self.diff = abs(self.user_a-self.beam_a)
        self.score = 100 - self.diff
        frame_text = self.font.render("Frame: {:d}".format(self.num_frame), True, self.COLOR_WHITE)
        self.screen.blit(frame_text,((self.GAME_WIDTH - frame_text.get_width()) // 2,(self.GAME_FLOOR + self.FONT_SIZE // 2)))
        score_text = self.font.render("Score: {:d}".format(self.score), True, self.COLOR_WHITE)
        self.screen.blit(score_text,((self.GAME_WIDTH - score_text.get_width()) // 2,(self.GAME_FLOOR + self.FONT_SIZE // 2 + self.FONT_SIZE)))
#        reward_text = self.font.render("Reward: {:.2f}".format(self.reward), True, self.COLOR_WHITE)
#        self.screen.blit(reward_text,((self.GAME_WIDTH - reward_text.get_width()) // 2,(self.GAME_FLOOR + self.FONT_SIZE // 2 + self.FONT_SIZE)))

        pygame.display.flip()
        self.screen.fill(self.COLOR_BLACK, (0, 0, self.GAME_WIDTH, self.GAME_HEIGHT))
        self.screen.blit(self.user, (self.user_x, self.user_y)) 
        self.screen.blit(self.beam, self.rect)

        self.last_diff = 0
        if self.num_frame == self.MAX_FRAME:               
            self.num_frame = 0
            if self.user_a == self.beam_a:
                self.reward += 1
            else:
                self.reward -= 1
            # get the last diff
            self.last_diff = abs(self.beam_a-self.user_a)
            self.user_a = random.randrange(30, 150, self.BEAM_VELOCITY)            
            self.user_r = math.radians(self.user_a)                              ###삼각함수는 라디안으로 써야됨
            self.user_x = round(self.USER_RADIUS*math.cos(self.user_r))+self.CENTER_x
            self.user_y = round(self.USER_RADIUS*math.sin(self.user_r))+self.CENTER_y
            self.game_over = True
        
        # save last 4 frames

        self.frames.append(pygame.surfarray.array2d(self.screen))                                                                   
        clock = pygame.time.Clock()
        clock.tick(30)                                                           ###take delay for 30 FPS (10,30,60,120...)
                                                            ###update whole screen                               

        return self.get_frames(), self.reward, self.game_over, self.num_frame, self.diff_av, self.first_diff, self.last_diff

    def get_frames(self):
        return np.array(list(self.frames))


if __name__ == "__main__":  
    game_1 = MyBeamforming_game_1()

    NUM_EPOCHS = 10
    for e in range(NUM_EPOCHS):
        print("Epoch: {:d}".format(e))
        game_1.reset()
        input_t = game_1.get_frames()
        game_over = False
        while not game_over:
            action = np.random.randint(0, 3, size=1)[0]
            input_tp1, reward, game_over, num_frame, diff_av, first_diff, last_diff = game_1.step(action, num_frame, reward)
            print(action, reward, game_over, num_frame, diff_av, first_diff, last_diff)
