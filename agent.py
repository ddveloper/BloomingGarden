''' Dawei Zhang - daweizhang009@gmail.com
    Solve flash game 'BloomingGarden' @ https://www.miniclip.com/games/bloomin-gardens/en
    ----------
    File agent.py - train it based on gym-bloomingGarden
'''
'''
code reference: https://github.com/GaetanJUVIN/Deep_QLearning_CartPole/blob/master/cartpole.py
'''

import gym
import gym_bloomingGarden
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import plot_model

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup = 'bloomingGarden_weight.h5'
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        # model = Sequential()
        # model.add(Conv2D(32, (3, 3), padding='same',
        #                 input_shape=self.state_size))
        # model.add(Activation('relu'))
        # model.add(Conv2D(32, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Conv2D(64, (3, 3), padding='same'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Conv2D(128, (3, 3), padding='same'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(128, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Flatten())
        # model.add(Dense(512))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(self.action_size))
        # model.add(Activation('linear'))
        
        # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
        self.model.save(self.weight_backup)

    def act(self, state, valid_acts):
        if np.random.rand() <= self.exploration_rate:
            return valid_acts[0] + valid_acts[1] * 9 + valid_acts[2] * 81 + valid_acts[3] * 81 * 9
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        act_id = 0
        for i in range(4): act_id += action[i] * (9 ** i)
        self.memory.append((state, act_id, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class BloomingG:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 10000
        self.env = gym.make('bloomingGarden-v0')
        self.state_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1] + 3
        self.action_size = 1
        for i in range(len(self.env.action_space)): self.action_size *= self.env.action_space[i].n
        self.agent = Agent(self.state_size, self.action_size)

    def run(self):
        try:
            cnt_500 = 0
            for i_episode in range(self.episodes):
                state = self.env.reset() / 8 + 0.001
                state = np.reshape(state, [1, self.state_size])
                done = False; score = 0
                valid_actions = (0,0,0,1)
                while not done:
                    #self.env.render()
                    action = self.agent.act(state, valid_actions)
                    action = self.convert_act(action)
                    next_state, reward, done, mv = self.env.step(action)
                    score = next_state[1]; valid_actions = mv.pop()
                    next_state = np.reshape(next_state[0] / 8 + 0.001, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                print(f"Episode {i_episode}# Score: {score}")
                self.agent.replay(self.sample_batch_size)
                if score == 500: cnt_500 += 1
                if cnt_500 > 50: break
        finally:
            self.agent.save_model()

    def convert_act(self, act_id):
        act = []
        for i in range(4):
            act.append(act_id % 9); act_id //= 9
        return (act[0], act[1], act[2], act[3])

    def play_back(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        done = False; score = 0
        valid_actions = (0,0,0,1)
        while not done:
            self.env.render()
            action = self.agent.act(state, valid_actions)
            action = self.convert_act(action)
            next_state, reward, done, mv = self.env.step(action)
            score = next_state[1]; valid_actions = mv.pop()
            next_state = np.reshape(next_state[0], [1, self.state_size])
            self.agent.remember(state, action, reward, next_state, done)
            state = next_state
        print(f"Playback # Score: {score}")

if __name__ == '__main__':
    train = True # switch between train & evaluate
    if train: 
        if os.path.isfile("bloomingGarden_weight.h5"):
            os.remove("bloomingGarden_weight.h5")
    game = BloomingG()
    if train: 
        game.run()
    else: game.play_back()
