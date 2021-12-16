from puzzle import *
from threading import Thread
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Input, merge
from keras.optimizers import Adam
import os
import cv2
from keras.models import Model

import random
from math import pow
NUM_ACTIONS = 4
NUM_STATES = 256

MAX_REPLAY_STATES = 10000
BATCH_SIZE = 64
NUM_GAMES_TRAIN = 1000

IMG_SIZE = 1024

background_color = [125, 135, 146]

margin_size = IMG_SIZE / 32

colors = {0: [138, 148, 158], 2: [218, 228, 238], 4: [200, 224, 237], 8: [121, 177, 242], 16: [99, 149, 245],
          32: [95, 124, 246], 64: [59, 94, 246], 128: [114, 207, 237], 256: [97, 204, 237],
          512: [237, 200, 80], 1024: [237, 197, 63], 2048: [46, 194, 237]}

num_colors = {0: [138, 148, 158], 2: [101, 110, 119], 4: [101,110, 119], 8: [255, 255, 255], 16: [255, 255, 255],
              32: [255, 255, 255], 64: [255, 255, 255], 128: [255, 255, 255], 256: [255, 255, 255], 512: [255, 255, 255],
              1024: [255, 255, 255], 2048: [255, 255, 255]}


def create_board(state):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    img[:] = background_color

    for i in range(0, 4):
        for j in range(0, 4):
            value = state[i * 4 + j]
            start_x = int(IMG_SIZE / 4 * j + margin_size)
            start_y = int(IMG_SIZE / 4 * i + margin_size)
            img = cv2.rectangle(img, pt1=(start_x, start_y),
                                pt2=(start_x + int(IMG_SIZE / 4 - 2 * margin_size), int(start_y + IMG_SIZE / 4 - 2 * margin_size)),
                                color=(colors[value][0], colors[value][1], colors[value][2]),
                                thickness=-1)
            font = cv2.FONT_HERSHEY_SIMPLEX

            value_str = str(value)
            dict_start_points = {1: 70, 2: 50, 3: 30, 4: 10}
            cv2.putText(img, str(value), (start_x + dict_start_points[len(value_str)], start_y + int( (IMG_SIZE / 4 - 2 * margin_size) / 2)), font, 2,
                        (num_colors[value][0], num_colors[value][1], num_colors[value][2]), 5, cv2.LINE_AA)
    return img



# 0 - UP, 1 - DOWN, 2 - LEFT, 3 - RIGHT


def create_conv_model(n_inputs, n_outputs):
    input_matrix = Input(shape=(16, 4, 4,))
    conv_a = Convolution2D(nb_filter=128,  nb_row=2, nb_col=1, border_mode="valid", activation="relu")(input_matrix)
    conv_b = Convolution2D(nb_filter=128,  nb_row=1, nb_col=2, border_mode="valid", activation="relu")(input_matrix)

    conv_aa = Convolution2D(nb_filter=1024,  nb_row=2, nb_col=1, border_mode="valid", activation="relu")(conv_a)
    conv_ab = Convolution2D(nb_filter=1024, nb_row=1, nb_col=2, border_mode="valid", activation="relu")(conv_a)

    conv_ba = Convolution2D(nb_filter=1024,  nb_row=2, nb_col=1, border_mode="valid", activation="relu")(conv_b)
    conv_bb = Convolution2D(nb_filter=1024, nb_row=1, nb_col=2, border_mode="valid", activation="relu")(conv_b)

    conv_a_flatten = Flatten()(conv_a)
    conv_b_flatten = Flatten()(conv_b)

    conv_aa_flatten = Flatten()(conv_aa)
    conv_ab_flatten = Flatten()(conv_ab)
    conv_ba_flatten = Flatten()(conv_ba)
    conv_bb_flatten = Flatten()(conv_bb)

    merged = merge([conv_a_flatten, conv_b_flatten, conv_aa_flatten, conv_ab_flatten, conv_ba_flatten, conv_bb_flatten],
                   mode="concat")

    dense_layer = Dense(4, activation="linear")(merged)

    model = Model(input=input_matrix, output=dense_layer)
    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='mse')
    return model

gamegrid = GameGrid()
model = create_conv_model(NUM_STATES, NUM_ACTIONS)


gamma = 0.99
epsilon = 0.8

# Initials


def transform_state_to_image(state):
    one_hot_state = np.zeros(shape=(16,16))

    for i in range(0, len(state)):
        for j in range(0,one_hot_state.shape[1]):
            if state[i] == 0:
                one_hot_state[j][i] = 1
                break
            if state[i] == pow(2, j):
                one_hot_state[j][i] = 1
                break

    res = np.reshape(one_hot_state, newshape=(len(state), 4, 4))

    return res

# Main game loop

for number_game in range(NUM_GAMES_TRAIN):
    replay = []
    gamegrid.init_matrix()
    new_state = gamegrid.matrix
    reward_game = 0
    done = False
    loss = 0
    index_train_per_game = 0
    print('[+] Starting game ' + str(number_game))

    while not done:
        index_train_per_game += 1
        if random.random() < epsilon:
            action = np.random.randint(NUM_ACTIONS)
        else:
            new_state_combined = np.asarray(sum(new_state, []))
            new_state_combined = transform_state_to_image(new_state_combined)
            q = model.predict(new_state_combined.reshape(1, 16,4,4))[0]
            action = np.argmax(q)

        old_state = new_state

        new_state, reward, action_number, done = gamegrid.key_down(action)

        new_state_combined = np.asarray(sum(new_state, []))
        board = create_board(new_state_combined)

        if done:
            cv2.putText(board, "STARTING TRAINING", (int(IMG_SIZE / 4), int(IMG_SIZE / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("2048", board)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        reward_game += reward

        replay.append([new_state, reward, action, done, old_state])

    X_train = np.zeros((len(replay), 16,4,4))
    Y_train = np.zeros((len(replay), NUM_ACTIONS))

    for index_rep in range(len(replay)):
        new_rep_state, reward_rep, action_rep, done_rep, old_rep_state = replay[index_rep]

        new_rep_state = np.asarray(sum(new_rep_state, []))
        old_rep_state = np.asarray(sum(old_rep_state, []))

        new_rep_state = transform_state_to_image(new_rep_state)
        old_rep_state = transform_state_to_image(old_rep_state)

        old_rep_state = old_rep_state.reshape(1, 16,4,4)
        new_rep_state = new_rep_state.reshape(1, 16,4,4)

        old_q = model.predict(old_rep_state)[0]
        new_q = model.predict(new_rep_state)[0]

        update_target = np.copy(old_q)
        if done_rep:
            update_target[action_rep] = reward_rep
        else:
            update_target[action_rep] = reward_rep + (gamma * np.max(new_q))

        X_train[index_rep] = old_rep_state
        Y_train[index_rep] = update_target

    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    model.fit(X_train, Y_train, batch_size=16, nb_epoch=1)


    print("[+] End Game {} | Reward {} | Epsilon {:.4f} | TrainPerGame {} ".format(number_game, reward_game,
                                                                                               epsilon,
                                                                                           index_train_per_game))

    # Print terminal state

    for row in new_state:
        print(row)
    if epsilon >= 0.01:
        epsilon -= 10 / NUM_GAMES_TRAIN
