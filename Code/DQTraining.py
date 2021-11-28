
import cv2
import numpy as np
def resize_frame(frame):
    frame = np.average(frame,axis = 2)
    frame = cv2.resize(frame,(84,84),interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame,dtype = np.uint8)
    return frame

"""MEMORY

---



---


"""

from collections import deque

class Memory():
    def __init__(self,max_len):
        self.max_len = max_len
        self.frames = deque(maxlen = max_len)
        self.actions = deque(maxlen = max_len)
        self.rewards = deque(maxlen = max_len)
        self.done_flags = deque(maxlen = max_len)

    def add_experience(self,next_frame, next_frames_reward, next_action, next_frame_terminal):
        self.frames.append(next_frame)
        self.actions.append(next_action)
        self.rewards.append(next_frames_reward)
        self.done_flags.append(next_frame_terminal)

"""**AGENT** (MODEL,GET ACTION, LEARN(STATE,NEW STATE))"""

from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
#import keras.backend as K
import tensorflow as tf
import numpy as np
import random


class Agent():
    def __init__(self,possible_actions,starting_mem_len,max_mem_len,starting_epsilon,learn_rate, debug = False):
        self.memory = Memory(max_mem_len)
        self.possible_actions = possible_actions
        self.epsilon = starting_epsilon
        self.epsilon_decay = .9/100000
        self.epsilon_min = .05
        self.gamma = .99
        self.learn_rate = learn_rate
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        self.starting_mem_len = starting_mem_len
        self.learns = 0


    def _build_model(self):
        model1 = Sequential()
        model1.add(Input((84,84,4)))
        model1.add(Conv2D(32,8,strides = 4,activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model1.add(Conv2D(64,4,strides = 2,activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model1.add(Conv2D(64,3,strides = 1,activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model1.add(Flatten())
        model1.add(Dense(512,activation = 'relu'))
        model1.add(Dense(len(self.possible_actions), activation = 'linear',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        optimizer = Adam(self.learn_rate)
        model1.compile(optimizer, loss=tf.keras.losses.Huber())
        print(model1.summary())
        print('\nAgent Initialized\n')
        return model1
    def _build_model2(self):
        model2 = Sequential()
        model2.add(Input((84,84,4)))
        model2.add(Conv2D(64,8,strides = 4,activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model2.add(Conv2D(64,4,strides = 2,activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model2.add(Conv2D(64,3,strides = 1,activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model2.add(Flatten())
        model2.add(Dense(512,activation = 'relu'))
        model2.add(Dense(len(self.possible_actions), activation = 'linear',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        optimizer = Adam(self.learn_rate)
        model2.compile(optimizer, loss=tf.keras.losses.Huber())
        print(model2.summary())
        print('\nAgent Initialized\n')
        return model2


    def get_action(self,token,state):

        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions,1)[0]
        if token == 0:
            a_index = np.argmax(self.model.predict(state))
        else:
            a_index = np.argmax(self.model_target.predict(state))

        return self.possible_actions[a_index]

    def _index_valid(self,index):
        if self.memory.done_flags[index-3] or self.memory.done_flags[index-2] or self.memory.done_flags[index-1] or self.memory.done_flags[index]:
            return False
        else:
            return True

    def learn(self,token):
        states = []
        next_states = []
        actions_taken = []
        next_rewards = []
        next_done_flags = []

        while len(states) < 32:
            index = np.random.randint(4,len(self.memory.frames) - 1)
            if self._index_valid(index):
                state = [self.memory.frames[index-3], self.memory.frames[index-2], self.memory.frames[index-1], self.memory.frames[index]]
                state = np.moveaxis(state,0,2)/255
                next_state = [self.memory.frames[index-2], self.memory.frames[index-1], self.memory.frames[index], self.memory.frames[index+1]]
                next_state = np.moveaxis(next_state,0,2)/255

                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index+1])
                next_done_flags.append(self.memory.done_flags[index+1])
        if token == 0:
            labels = self.model.predict(np.array(states))
            next_state_values = self.model_target.predict(np.array(next_states))
        else:
            next_state_values = self.model.predict(np.array(states))
            labels = self.model_target.predict(np.array(next_states))
        
       
        for i in range(32):
            action = self.possible_actions.index(actions_taken[i])
            labels[i][action] = next_rewards[i] + (not next_done_flags[i]) * self.gamma * max(next_state_values[i])
        self.model.fit(np.array(states),labels,batch_size = 32, epochs = 1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1
        # if self.learns % 500 == 0:
        #     self.model_target.set_weights(self.model.get_weights())
        #     print('\nTarget model updated')

"""NEW GAME"""

import gym
import numpy as np


def initialize_new_game(name, env, agent):  
    env.reset()
    starting_frame = resize_frame(env.step(0)[0])

    dummy_action = 0
    dummy_reward = 0
    dummy_done = False
    for i in range(3):
        agent.memory.add_experience(starting_frame, dummy_reward, dummy_action, dummy_done)

def make_env(name, agent):
    env = gym.make(name)
    return env

"""**Take_Step**"""

def take_step(name, env, agent, score):
    token = 0
    
    #1 and 2: Update timesteps and save weights
    agent.total_timesteps += 1
    if agent.total_timesteps % 500 == 0:
      agent.model.save('DQLMOdel.hdf5')
      print('\nWeights saved!')

    #3: Take action
    next_frame, next_frames_reward, next_frame_terminal, info = env.step(agent.memory.actions[-1])
    
    #4: Get next state
    next_frame = resize_frame(next_frame)
    new_state = [agent.memory.frames[-3], agent.memory.frames[-2], agent.memory.frames[-1], next_frame]
    new_state = np.moveaxis(new_state,0,2)/255 #We have to do this to get it into keras's goofy format of [batch_size,rows,columns,channels]
    new_state = np.expand_dims(new_state,0) #^^^
    
    #5: Get next action, using next state
    next_action = agent.get_action(token,new_state)

    #6: If game is over, return the score
    if next_frame_terminal:
        agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)
        return (score + next_frames_reward),True

    #7: Now we add the next experience to memory
    agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)

    #8: If we are trying to debug this then render
    #if debug:
       # env.render()

    #9: If the threshold memory is satisfied, make the agent learn from memory
    if len(agent.memory.frames) > agent.starting_mem_len:
        agent.learn(token)
    if agent.learns % 500 == 0 and token == 0:
      token = 1
      
    elif agent.learns % 500 == 0 and token == 1:
      token = 0
    # print(token)
    

    return (score + next_frames_reward),False

def play_episode(name, env, agent, debug = False):
    initialize_new_game(name, env, agent)
    done = False
    score = 0
    while True:
        score,done = take_step(name,env,agent,score)
        if done:
            break
    return score

import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np

name = 'PongDeterministic-v4'

agent = Agent(possible_actions=[0,2,3],starting_mem_len=15000,max_mem_len=20000,starting_epsilon = 1, learn_rate = .00025)
env = make_env(name,agent)

last_50_avg = [-21]
scores = deque(maxlen = 50)
max_score = -21

""" If testing:
agent.model.load_weights('recent_weights.hdf5')
agent.model_target.load_weights('recent_weights.hdf5')
agent.epsilon = 0.0
"""

env.reset()

for i in range(1000):
    timesteps = agent.total_timesteps
    timee = time.time()
    score = play_episode(name, env, agent) #set debug to true for rendering
    scores.append(score)
    if score > max_score:# experience replay buffer, 50000 and 
        max_score = score

    print('\nEpisode: ' + str(i))
    print('Steps: ' + str(agent.total_timesteps - timesteps))
    print('Duration: ' + str(time.time() - timee))
    print('Score: ' + str(score))
    print('Max Score: ' + str(max_score))
    print('Epsilon: ' + str(agent.epsilon))

    if i%50==0 and i!=0:
        last_50_avg.append(sum(scores)/len(scores))
        plt.plot(np.arange(0,i+1,50),last_50_avg)
        plt.show()
if max_score == 10:
        break



