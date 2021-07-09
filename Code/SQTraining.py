#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
def resize_frame(frame):
    
    frame = np.average(frame,axis = 2)
    frame = cv2.resize(frame,(84,84),interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame,dtype = np.uint8)
    return frame


# In[10]:


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


# In[11]:


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
      
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions,1)[0]

        a_index = np.argmax(self.model.predict(state))
        return self.possible_actions[a_index]

    def _index_valid(self,index):
        if self.memory.done_flags[index-3] or self.memory.done_flags[index-2] or self.memory.done_flags[index-1] or self.memory.done_flags[index]:
            return False
        else:
            return True

    def learn(self,debug = False):
       
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

        
        labels = self.model.predict(np.array(states))
        next_state_values = self.model_target.predict(np.array(next_states))
        
        
        for i in range(32):
            action = self.possible_actions.index(actions_taken[i])
            labels[i][action] = next_rewards[i] + (not next_done_flags[i]) * self.gamma * max(next_state_values[i])

        
        self.model.fit(np.array(states),labels,batch_size = 32, epochs = 1, verbose = 0)

       
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1
        
       
        if self.learns % 1000 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')


# In[12]:


import gym
import numpy as np
def make_env(name, agent):
    env = gym.make(name)
    return env


# In[13]:


def init_new_game(name, env, agent):
    
    env.reset()
    starting_frame = resize_frame(env.step(0)[0])

    dummy_action = 0
    dummy_reward = 0
    dummy_done = False
    for i in range(3):
        agent.memory.add_experience(starting_frame, dummy_reward, dummy_action, dummy_done)


# In[14]:



def take_step(name, env, agent, score, debug):
    
    #1 and 2: Update timesteps and save weights
    agent.total_timesteps += 1
    if agent.total_timesteps % 50000 == 0:
      agent.model.save_weights('SingleQL_weights.hdf5')
      agent.model.save('SingleQL.hdf5')
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

    #7: Now we add the next experience to memory
    agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)

    #8: If we are trying to debug this then render
    #if debug:
       # env.render()

    #9: If the threshold memory is satisfied, make the agent learn from memory
    if len(agent.memory.frames) > agent.starting_mem_len:
        agent.learn(debug)

    return (score + next_frames_reward),False


# In[15]:


def play_episode(name, env, agent, debug = False):
    init_new_game(name, env, agent)
    done = False
    score = 0
    while True:
        score,done = take_step(name,env,agent,score, debug)
        if done:
            break
    return score


# In[ ]:


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
    score = play_episode(name, env, agent, debug = False)#set debug to true for rendering
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
    if max_score == 21:
        break


# In[1]:


print("Agent Won")


# In[ ]:


mem = np.asarray(agent.memory)
np.save('Memory.npy', mem)


# In[ ]:


f = np.load('C:\\Users\\madhu\\Downloads\\Memory.npy', allow_pickle=True)
f.shape


# In[ ]:


print(f)


# In[ ]:


print(mem)


# In[ ]:


np.save()


# In[ ]:


fr = np.asarray(agent.memory.frames)
ac = np.asarray(agent.memory.actions)
re = np.asarray(agent.memory.rewards)
fl = np.asarray(agent.memory.done_flags)


# In[ ]:


np.save('frames.npy',fr)
np.save('actions.npy',ac)
np.save('rewards.npy',re)
np.save('flags.npy',fl)


# In[ ]:


print("Complete")


# In[ ]:


agent.model.save('modelbackup.hdf5')


# In[ ]:




