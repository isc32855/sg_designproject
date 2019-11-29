# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from scipy.misc import imresize
import collections
import numpy as np
import PIL
import os
import math
import ddqn_game_move


def preprocess_images(images):
    if images.shape[0] < 4:
        # single image
        x_t = images[0]
        x_t = np.array(PIL.Image.fromarray(x_t).resize((40, 40)).convert("L"))
        x_t = x_t[0:40,0:20].copy()
        x_t = x_t.astype("float")
        x_t /= 255.0                                                    
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)                  
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
    s_t = np.expand_dims(s_t, axis=0)                                    
    return s_t                                                     
                                                        
def get_next_batch(experience, target_model, model, num_actions, gamma, batch_size):
    batch_indices = np.random.randint(low=0, high=len(experience),
                                      size=batch_size)
    batch = [experience[i] for i in batch_indices]
    X = np.zeros((batch_size, 40, 20, 4))
    Y = np.zeros((batch_size, num_actions))
    Y_tmp = np.zeros((batch_size, num_actions)) 
    for i in range(len(batch)):
        s_t, a_t, r_t, s_tp1, game_over = batch[i]
        X[i] = s_t
        Y[i] = model.predict(s_t)[0]
        f_a_t = np.argmax(model.predict(s_tp1)[0])
        Q_sa = target_model.predict(s_tp1)[0][f_a_t]
        if game_over:
            Y[i, a_t] = r_t
        else:
            Y[i, a_t] = r_t + gamma * Q_sa
    return X, Y
############################# main ###############################

# initialize parameters
DATA_DIR = "C:/Users/82103/Desktop/DesignProject/jupyter/bf"
NUM_ACTIONS = 3 # number of valid actions (left, stay, right)
GAMMA = 0.99 # decay rate of past observations
INITIAL_EPSILON = 0.1 # starting value of epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
MEMORY_SIZE = 50000 # number of previous transitions to remember
NUM_EPOCHS_OBSERVE = 1000
NUM_EPOCHS_TRAIN = 0

BATCH_SIZE = 32                                 
NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN

# build the model                                     
model = Sequential()
model.add(Conv2D(16, kernel_size=4, strides=2,   
                 kernel_initializer="normal", 
                 padding="same",
                 input_shape=(40, 20, 4)))
model.add(Activation("relu"))
model.add(Conv2D(32, kernel_size=2, strides=1, 
                 kernel_initializer="normal", 
                 padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(32, kernel_size=2, strides=1, 
                 kernel_initializer="normal",
                 padding="same"))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(256, kernel_initializer="normal"))
model.add(Activation("relu"))
model.add(Dense(3, kernel_initializer="normal"))
#model.load_weights('')
model.compile(optimizer=Adam(lr=1e-6), loss="mse")

#target model
target_model = Sequential()
target_model.add(Conv2D(16, kernel_size=4, strides=2,  
                 kernel_initializer="normal", 
                 padding="same",
                 input_shape=(40, 20, 4)))
target_model.add(Activation("relu"))
target_model.add(Conv2D(32, kernel_size=2, strides=1, 
                 kernel_initializer="normal", 
                 padding="same"))
target_model.add(Activation("relu"))
target_model.add(Conv2D(32, kernel_size=2, strides=1, 
                 kernel_initializer="normal",
                 padding="same"))
target_model.add(Activation("relu"))
target_model.add(Flatten())
target_model.add(Dense(256, kernel_initializer="normal"))
target_model.add(Activation("relu"))
target_model.add(Dense(3, kernel_initializer="normal"))
#target_model.load_weights('easy_model_stop.h5')
target_model.compile(optimizer=Adam(lr=1e-6), loss="mse") 

# train network
game_1 = ddqn_game_move.MyBeamforming_game_1()                                 
experience = collections.deque(maxlen=MEMORY_SIZE)             

#fout = open(os.path.join(DATA_DIR, "ddqn_easy_results_move.tsv"), "wb")
num_games, num_wins = 0, 0
epsilon = INITIAL_EPSILON
for e in range(NUM_EPOCHS):                                  
    loss = 0.0
    game_1.reset()                                                      
    
    # get first state
    a_0 = 0  
    f_0 = 0  
    r_0 = 0
    x_t, r_t, game_over, f_0, dav, fd, ld = game_1.step(a_0,f_0,r_0)                           
    s_t = preprocess_images(x_t)

    while not game_over:
        s_tm1 = s_t
        # next action
        if e <= NUM_EPOCHS_OBSERVE:
            a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0] 
        else:
            if np.random.rand() <= epsilon:                                
                a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
            else:
                q = model.predict(s_t)[0]
                a_t = np.argmax(q)                                      
                
                q_array.append(q)
                a_t_array.append(a_t)
        # apply action, get reward
        x_t, r_t, game_over, f_0, dav, fd, ld = game_1.step(a_t, f_0, r_t)
        s_t = preprocess_images(x_t)

        # store experience
        experience.append((s_tm1, a_t, r_t, s_t, game_over))
        
        if e > NUM_EPOCHS_OBSERVE:
            # finished observing, now start training
            # get next batch
            X, Y = get_next_batch(experience, target_model, model, NUM_ACTIONS, 
                                  GAMMA, BATCH_SIZE)
            
            loss += model.train_on_batch(X, Y) 

    ### update target net
    if e % 50 == 0:
        target_model.set_weights(model.get_weights())

    # if exactly matched, increment num_wins
    if ld == 0:
        num_wins += 1      
    ### reduce epsilon gradually
    if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPOCHS
    
#    print("Epoch %4d/%d | Loss %.5f | Win Count: %d | reward : %.2f | dav : %.2f | fd : %d | ld : %d"
#          %(e + 1, NUM_EPOCHS, loss, num_wins, r_t, dav, fd, ld))

#    fout.write(b"%4d\t%.5f\t%d\t%d\t%.2f\t%d\t%d\n"
#          %(e + 1, loss, num_wins, r_t, dav, fd, ld))
#    if e % 100 == 0:                                                      ###100 epoch마다 save
#        model.save(os.path.join(DATA_DIR, "ddqn_easy_model_move.h5"), overwrite=True)
    
#fout.close()
#model.save(os.path.join(DATA_DIR, "ddqn_easy_model_move.h5"), overwrite=True)


