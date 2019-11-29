#!/usr/bin/env python
# coding: utf-8
# Import a library of functions called 'pygame'
from __future__ import division, print_function
import pygame
import math
import random
import time
import os
DATA_DIR = "C:/Users/82103/Desktop/DesignProject/jupyter/keyboard"
pygame.init()
# set constants
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (64, 255, 64)
GAME_WIDTH = 400
GAME_HEIGHT = 400
GAME_FLOOR = 300
USER_WIDTH = 30
USER_HEIGHT = 30
USER_RADIUS = 200
BEAM_WIDTH = 150
BEAM_HEIGHT = 30
CENTER_x = 200
CENTER_y = 200
USER_VELOCITY = 30                                                  ### 32 flames -> 160
BEAM_VELOCITY = 30
FONT_SIZE = 30
CUSTOM_EVENT = pygame.USEREVENT + 1
font = 0
font = pygame.font.SysFont("Comic Sans MS", FONT_SIZE)
MAX_FRAME = 32

done = False
flag = None
win_flag = None
num_frame = 0
win_count = 0
clock= pygame.time.Clock()

# initialize positions
beam_a = 90
beam_r = math.radians(beam_a)
beam_x = -0.5*BEAM_HEIGHT*math.sin(beam_r)+CENTER_x
beam_y = -0.5*BEAM_HEIGHT*math.cos(beam_r)+CENTER_y

user_a = random.randrange(30, 150, USER_VELOCITY)
user_r = math.radians(user_a)
user_x = round(USER_RADIUS*math.cos(user_r))+CENTER_x
user_y = -round(USER_RADIUS*math.sin(user_r))+CENTER_y

# set up display, clock, etc
screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
pygame.display.set_caption('DTL-Design Project')


while not done:
    if win_count <= 3:
        pass
    elif win_count > 3 and win_count <=6:
        USER_VELOCITY = 15
        BEAM_VELOCITY = 15
    elif win_count > 6 and win_count <=9:
        USER_VELOCITY = 10
        BEAM_VELOCITY = 10
    else:
        USER_VELOCITY = 5
        BEAM_VELOCITY = 5
    # This limits the while loop to a max of 10 times per second.
    # Leave this out and we will use all CPU we can.
    clock.tick(30)
    # Main Event Loop
    for event in pygame.event.get():# User did something
        if event.type == pygame.KEYDOWN:# If user release what he pressed.
            pressed= pygame.key.get_pressed()
            buttons= [pygame.key.name(k)for k,v in enumerate(pressed) if v]
            flag= True
        elif event.type == pygame.KEYUP:# If user press any key.
            flag= False
        elif event.type == pygame.QUIT: # If user clicked close.
            done= True

    if flag == True:
        if buttons[0] == 'a':
            #move left
            if beam_a >= 150:
                beam_a = 150
            else:
                 beam_a += BEAM_VELOCITY
            beam_r = math.radians(beam_a)
            beam_x = -0.5*BEAM_HEIGHT*math.sin(beam_r)+CENTER_x
            beam_y = -0.5*BEAM_HEIGHT*math.cos(beam_r)+CENTER_y
        elif buttons[0] == 'd':
            #move right
            if beam_a <= 30:
                beam_a = 30
            else:
                beam_a -= BEAM_VELOCITY
            beam_r = math.radians(beam_a)
            beam_x = -0.5*BEAM_HEIGHT*math.sin(beam_r)+CENTER_x
            beam_y = -0.5*BEAM_HEIGHT*math.cos(beam_r)+CENTER_y
        else:
            pass
    elif flag== False:
        num_frame += 1
    else:
        pass


    if num_frame % 4 == 0 and num_frame < 32:
        user_action = random.randint(0, 9)
    else:
        user_action = 2

    if user_action == 0:   ### move user left
        if user_a >= 150:
            user_a = 150
        else:
            user_a += USER_VELOCITY
        user_r = math.radians(user_a)
        user_x = round(USER_RADIUS*math.cos(user_r))+CENTER_x
        user_y = -round(USER_RADIUS*math.sin(user_r))+CENTER_y
    elif user_action == 1:  ### move user right
        if user_a <= 30:
            user_a = 30
        else:
            user_a -= USER_VELOCITY
        user_r = math.radians(user_a)
        user_x = round(USER_RADIUS*math.cos(user_r))+CENTER_x
        user_y = -round(USER_RADIUS*math.sin(user_r))+CENTER_y
    else:
        pass


    ld = abs(beam_a - user_a)
    score = 100 - ld

    screen.fill(COLOR_GREEN, (0, 0, GAME_WIDTH, GAME_HEIGHT))

    city = pygame.image.load(os.path.join(DATA_DIR,'crossroad.png'))
    city = pygame.transform.scale(city, (GAME_WIDTH, GAME_HEIGHT))
    screen.blit(city, (0, 0))

    # draw City
    building = pygame.image.load(os.path.join(DATA_DIR,'building.png'))
    building = pygame.transform.scale(building, (GAME_WIDTH//8, GAME_HEIGHT//4))
    screen.blit(building, (CENTER_x - GAME_WIDTH//16, GAME_HEIGHT//2+10))

    # update user position
    pygame.draw.rect(screen, COLOR_WHITE,[user_x, user_y,USER_WIDTH,USER_HEIGHT])
    user = pygame.image.load(os.path.join(DATA_DIR,'smartphone.png'))
    user = pygame.transform.scale(user, (USER_WIDTH,USER_HEIGHT))
    screen.blit(user, (user_x, user_y))

    # update beam position
    original_beam = pygame.image.load(os.path.join(DATA_DIR,'ellipse.png'))
    original_beam = pygame.transform.scale(original_beam, (BEAM_WIDTH,BEAM_HEIGHT))
    beam = pygame.transform.rotate(original_beam, beam_a)
    rect = beam.get_rect()
    rect.center = (CENTER_x + (BEAM_WIDTH/2)*math.cos(beam_r),
                        CENTER_y - (BEAM_WIDTH/2)*math.sin(beam_r))
    screen.blit(beam, rect)

    # draw Antenna
    pygame.draw.polygon(screen, COLOR_WHITE, [[CENTER_x + 10,CENTER_y],
                                                        [CENTER_x - 10,CENTER_y],
                                                        [CENTER_x, CENTER_y + 10]],1)
    pygame.draw.lines(screen, COLOR_WHITE, False, [[CENTER_x,CENTER_y],
                                                             [CENTER_x,CENTER_y + 25],
                                                             [CENTER_x + 10,CENTER_y + 25]],1)
    ld = abs(user_a-beam_a)
    score = 100 - ld
    frame_text = font.render("Frame: {:d}".format(num_frame), True, COLOR_WHITE)
    screen.blit(frame_text,((GAME_WIDTH - frame_text.get_width()) // 2,(GAME_FLOOR + FONT_SIZE // 2 )))
    score_text = font.render("Score: {:d}".format(score), True, COLOR_WHITE)
#    screen.blit(score_text,((GAME_WIDTH - score_text.get_width()) // 2,(GAME_FLOOR + FONT_SIZE // 2 )))
    win_text = font.render("win: {:d}".format(win_count), True, COLOR_WHITE)
    screen.blit(win_text,((GAME_WIDTH - win_text.get_width()) // 2,(GAME_FLOOR + FONT_SIZE // 2 + FONT_SIZE)))


    pygame.display.flip()


    if num_frame == 32:
        if ld == 0:
            num_frame = 0
            beam_a = 90
            beam_r = math.radians(beam_a)
            beam_x = -0.5*BEAM_HEIGHT*math.sin(beam_r)+CENTER_x
            beam_y = -0.5*BEAM_HEIGHT*math.cos(beam_r)+CENTER_y

            user_a = random.randrange(30, 150, USER_VELOCITY)
            user_r = math.radians(user_a)
            user_x = round(USER_RADIUS*math.cos(user_r))+CENTER_x
            user_y = -round(USER_RADIUS*math.sin(user_r))+CENTER_y

            win_count += 1
        else:
            done = True

# Be IDLE friendly
text = font.render("GAME OVER", True, COLOR_BLACK)
screen.blit(text,((GAME_WIDTH - text.get_width()) // 2,(GAME_HEIGHT // 2)))
pygame.display.flip()
time.sleep(5)
pygame.quit()
