# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imresize
import numpy as np
import os
import PIL
import ddqn_game_move

def preprocess_images(images):
    ### game의 첫부분에는 4개가 없을 수도 있으니깐 fake로 똑같은 놈(x_t)을 4번 넣어서 만들어준다
    if images.shape[0] < 4:
        # single image
        x_t = images[0]
        x_t = np.array(PIL.Image.fromarray(x_t).resize((40, 40)).convert("L"))         ### error때메 PIL 모듈 메서드로 바꿈
        x_t = x_t[0:40,0:20].copy()
        x_t = x_t.astype("float")
        x_t /= 255.0                                                      ### 흰색(0)~검은색(255)로 표현된 이미지 데이터를 [0,1]로 정규화
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)                      ### axis=2 라서 80x80 -> 80x80x4
    else:
        # 4 images
        xt_list = []
        for i in range(images.shape[0]):
            x_t = np.array(PIL.Image.fromarray(images[i]).resize((40, 40)).convert("L"))
            x_t = x_t[0:40,0:20].copy()
            x_t = x_t.astype("float")
            x_t /= 255.0
            xt_list.append(x_t)
        s_t = np.stack((xt_list[0], xt_list[1], xt_list[2], xt_list[3]), 
                       axis=2)
    s_t = np.expand_dims(s_t, axis=0)                                    ### 어렵다ㅠ
    return s_t 

############################# main ###############################

DATA_DIR = "C:/Users/82103/Desktop/DesignProject/jupyter/bf"

BATCH_SIZE = 32
NUM_EPOCHS = 100

model = load_model(os.path.join(DATA_DIR, "color_model_move5.h5"))
model.compile(optimizer=Adam(lr=1e-6), loss="mse")

# train network
game_1 = color_game_move.MyBeamforming_game_1()

num_games, num_wins = 0, 0
for e in range(NUM_EPOCHS):
    loss = 0.0
    game_1.reset()
    
    # get first state

    a_0 = 0 
    f_0 = 0  
    r_0 = 0
    x_t, r_t, game_over, f_0, dav,fd , ld = game_1.step(a_0,f_0,r_0)             
    s_t = preprocess_images(x_t)
    
    while not game_over:
        s_tm1 = s_t
        # next action
        q = model.predict(s_t)[0]
        a_t = np.argmax(q)
        # apply action, get reward
        x_t, r_t, game_over, f_0, dav,fd ,ld = game_1.step(a_t, f_0, r_t)
        s_t = preprocess_images(x_t)
        # if reward, increment num_wins
        if r_t == 1:
            num_wins += 1

    num_games += 1
    print("Game: {:03d}, Wins: {:03d}".format(num_games, num_wins), end="\r")
        
print("")


# In[ ]:




