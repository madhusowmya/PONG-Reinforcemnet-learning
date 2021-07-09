#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import numpy as np
def resize_frame(frame):
    
    frame = np.average(frame,axis = 2)
    frame = cv2.resize(frame,(84,84),interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame,dtype = np.uint8)
    return frame


# In[8]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:





# In[9]:


frames = np.load('C:\\Users\\madhu\\Desktop\\Deep Learning\\project 3\\RL\\frames.npy',allow_pickle=True)#test data laodingf
frames.shape


# In[10]:


actions = np.load('C:\\Users\\madhu\\Desktop\\Deep Learning\\project 3\\RL\\actions.npy',allow_pickle=True)#test data laoding
actions.shape


# In[ ]:





# In[11]:


rewards = np.load('C:\\Users\\madhu\\Desktop\\Deep Learning\\project 3\\RL\\rewards.npy',allow_pickle=True)#test data laoding
rewards.shape


# In[12]:


flags = np.load('C:\\Users\\madhu\\Desktop\\Deep Learning\\project 3\\RL\\flags.npy',allow_pickle=True)#test data laoding
flags.shape


# In[ ]:





# In[13]:


mem = np.load('C:\\Users\\madhu\\Desktop\\Deep Learning\\project 3\\RL\\Memory.npy', allow_pickle = True)#test data laoding
mem.shape


# In[20]:


mem = mem.tolist()


# In[21]:


from collections import deque

class Memory():
    def __init__(self,max_len):
        self.max_len = max_len
        self.frames = deque(frames)
        self.actions = deque(actions)
        self.rewards = deque(rewards)
        self.done_flags = deque(flags)

    def add_experience(self,next_frame, next_frames_reward, next_action, next_frame_terminal):
        self.frames.append(next_frame)
        self.actions.append(next_action)
        self.rewards.append(next_frames_reward)
        self.done_flags.append(next_frame_terminal)


# In[ ]:





# In[25]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
#import keras.backend as K

import numpy as np
import random


class Agent():

  def __init__(self,possible_actions,starting_mem_len,max_mem_len,starting_epsilon,learn_rate, debug = True):
        self.memory = mem
        self.possible_actions = possible_actions
        self.epsilon = starting_epsilon
        self.epsilon_decay = .9/100000
        self.epsilon_min = .05
        self.gamma = .99
        self.learn_rate = learn_rate
        self.model = self._build_model()
        self.model = load_model('C:\\Users\\madhu\\Desktop\\Deep Learning\\project 3\\RL\\SingleQL.hdf5')#test data laoding
        self.total_timesteps = 0
        self.starting_mem_len = starting_mem_len
        self.learns = 0

  def _build_model(self):
        model = Sequential()
        model.add(Input((84,84,4)))
        model.add(Conv2D(32,8,strides = 4,activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(64,4,strides = 2,activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(64,3,strides = 1,activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Flatten())
        model.add(Dense(512,activation = 'relu'))
        model.add(Dense(len(self.possible_actions), activation = 'linear',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        optimizer = Adam(self.learn_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        print('\nAgent Initialized\n')
        return model

    
  def get_action(self,state):
        """Explore"""
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions,1)[0]
        else:
            a_index = np.argmax(agent.model.predict(state))
            return self.possible_actions[a_index]

  def _index_valid(self,index):
        if self.memory.done_flags[index-3] or self.memory.done_flags[index-2] or self.memory.done_flags[index-1] or self.memory.done_flags[index]:
            return False
        else:
            return True


# In[ ]:





# In[26]:


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

def take_step(name, env, agent, score, debug):
    
    #1 and 2: Update timesteps and save model
    agent.total_timesteps += 1
    if agent.total_timesteps % 50000 == 0:
      # agent.model.save_weights('recent_weights.hdf5')
      print('\nWeights saved!')

    #3: Take action
    next_frame, next_frames_reward, next_frame_terminal, info = env.step(agent.memory.actions[-1])
    
    #4: Get next state
    next_frame = resize_frame(next_frame)
    new_state = [agent.memory.frames[-3], agent.memory.frames[-2], agent.memory.frames[-1], next_frame]
    new_state = np.moveaxis(new_state,0,2)/255 #We have to do this to get it into keras's goofy format of [batch_size,rows,columns,channels]
    new_state = np.expand_dims(new_state,0) #^^^
    
    #5: Get next action, using next state
    next_action = agent.get_action(new_state)

    #6: If game is over, return the score
    if next_frame_terminal:
        agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)
        return (score + next_frames_reward),True

    #7: add the next experience to memory
    agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)

    #8: If we are trying to debug this then render
    if debug:
       env.render()

    # #9: If the threshold memory is satisfied, make the agent learn from memory
    # if len(agent.memory.frames) > agent.starting_mem_len:
    #     agent.learn(debug)

    return (score + next_frames_reward),False

def play_episode(name, env, agent, debug = False):
    initialize_new_game(name, env, agent)
    done = False
    score = 0
    while True:
        score,done = take_step(name,env,agent,score, debug)
        if agent.epsilon > agent.epsilon_min:
          agent.epsilon -= agent.epsilon_decay
          agent.learns += 1
        
        if done:
            break
    return score


# In[27]:


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

env.reset()

for i in range(1000):
    timesteps = agent.total_timesteps
    timee = time.time()
    score = play_episode(name, env, agent, debug = False) 
   
    if score > max_score: 
        max_score = score

    print('\nEpisode: ' + str(i))
    print('Steps: ' + str(agent.total_timesteps - timesteps))
    print('Duration: ' + str(time.time() - timee))
    print('Score: ' + str(score))
    print('Max Score: ' + str(max_score))
    print('Epsilon: ' + str(agent.epsilon))

    if max_score == 21:
        break


# In[ ]:





# In[ ]:



  

