import gym
#import simpy

from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import  DummyVecEnv
#from stable_baselines.common.misc_util import flatten_action_mask
from stable_baselines import A2C, ACER, PPO2
from collections import defaultdict
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd
import random


class REEMAP(Env):
    def __init__(self):
        self.num_server=10
        self.action_space = Discrete(10)
        self.taken_server=-1
        self.num_edge_cloud=3
        self.num_regional_cloud=1
        self.num_edge_servers=[2,2,2]
        self.tot_edge_servers=6
        self.edge_cap=[150,150,150,150,150,150]
        self.num_regional_servers=4
        self.regional_cap=[150,150,150,150]
        self.midhaul=[10,10,10]
        
        self.observation_space = Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), high=np.array([250,250,250,250,250,250,250,250,250,250,10,10,10,0,0,0,0,0]))
        self.state = np.array([250,250,250,250,250,250,250,250,250,250,10,10,10,0,0,0,0,0])

        self.df = pd.read_csv('/content/drive/My Drive/dataset/telecom_italia_friday1sthour_9cell_sliced.csv')
        self.df = self.df.reset_index()  
        self.step_valid=0
    #    self.prevplacement=[[0 for i in range(0,4)]for j in range(0,num_cell)]
        self.curr_placement=[[[-1 for i in range(2)]for j in range(2)]for k in range(9)]
        self.df_index=0
        self.action_taken=-1
        self.timesteps=0
        self.exp_length = 36
        self.f=1
        self.tot_cost=0
        self.is_valid=0
        self.nodecost=[1,1,1,1,2,2,2,2,2,2]
        self.matrix=self.matrix1()
        print(self.matrix)
            
    def step(self, action):
        self.timesteps+=1
        reward=0
        cid=int(self.state[14])
        fid=int(self.state[15])
        sid=int(self.state[16])
        reward=0
        energy_cost=0
        self.is_valid=0
        c=0
        if self.state[action]>=self.state[17] and self.matrix[cid-1][sid][fid][action]==1: #Take action accordingly,check_validity
            
            self.is_valid=1
            if self.state[action]==200:
                self.tot_cost= self.tot_cost+100
                c=c+100
            self.state[action]=self.state[action]-self.state[17]
            energy_cost= self.state[17]*self.nodecost[action]
            #print("cid {}, fid {}, sid {}".format(cid,fid,sid))
            self.curr_placement[cid-1][sid][fid]=action
            # if self.f!=1 and prev_placement[cid][fid]!=action:
            #     self.migration_cost=self.state[13]*1# C
            # else: 
            #    self.f=0
            self.tot_cost=self.tot_cost+energy_cost#+self.migration_cost
            c=c+energy_cost
            if self.is_valid and fid==1 and sid==1:   #if action is valid and it is the last function of the base station
                reward= 1000-self.tot_cost
                self.tot_cost=0
        # else:                                   #else, reward=0 and skip to the next base station
        #     reward=0
        
        if self.is_valid==0:                   #skip to the next basestation
            while self.df_index<36 and self.df['Cid'].values[self.df_index]==cid :
                self.df_index+=1
            self.tot_cost=0
        else:
            self.df_index+=1
        # print("Action is {} tot icost is {} energy cost is {}".format(action, self.tot_cost,energy_cost))
        # print("Reward is {}".format(reward))

        self.exp_length -= 1 
        if self.exp_length == 0 or self.df_index>=36:               # Check if episode is done
            done = True
        else:
            done = False
        if done==False:
            k1=self.df['Time'].values[self.df_index]
            k2=self.df['Cid'].values[self.df_index]
            k3=self.df['Fid'].values[self.df_index]
            k4=self.df['Stype'].values[self.df_index]
            k5=self.df['Dem'].values[self.df_index]

            # if k1!=self.state[13]:
            #     self.state = [300,300,300,300,300,300,300,300,300,300,10,10,10,0,0,0,0,0]

            self.state[13]=k1
            self.state[14]=k2
            self.state[15]=k3
            self.state[16]=k4
            self.state[17]=k5

        
        
        info = {}
  #      print('obs=', self.state, 'action=', action, 'reward=', reward, 'done=', done)
        # Return step information
        action_mask = self.compute_action_mask(done)
        return self.state, reward, done, {'action_mask': action_mask,'place':self.curr_placement, 'tot_cost': self.tot_cost, 'c': c}

        
    def render(self, mode='human', **kwargs):
        print ('State: ' + repr(self.state) + ' Action: ' + repr(self.action_taken) + '\n')
    

    def reset(self):
        # Reset shower temperature
        self.state = [250,250,250,250,250,250,250,250,250,250,10,10,10,0,0,0,0,0]
        self.curr_placement=[[[-1 for i in range(2)]for j in range(2)]for k in range(9)]
        self.timesteps=0
        self.exp_length = 36
        self.action_taken=-1
        self.timesteps=0
        self.step_valid=0
        self.df_index=0
        self.tot_cost=0
      #  print("df index is{}".format(self.df_index))
        
        # Get a row from pandas dataframe
        # read and save it in the state 

        k1=self.df['Time'].values[self.df_index]
        k2=self.df['Cid'].values[self.df_index]
        k3=self.df['Fid'].values[self.df_index]
        k4=self.df['Stype'].values[self.df_index]
        k5=self.df['Dem'].values[self.df_index]

        self.state[13]=k1
        self.state[14]=k2
        self.state[15]=k3
        self.state[16]=k4
        self.state[17]=k5
        return self.state

    def matrix1(self):
        num_cell=9
        num_slice=2
        num_func=2
        num_nodes=10
        matrix=[[[[1 for l in range(num_nodes)]for k in range(num_func)]for j in range(num_slice)]for i in range(num_cell)]

        for i in range(0,num_cell):
          for j in range(0, num_slice):
              for l in range(0,4):
                  matrix[i][j][0][l]=0

        for i in range(0,num_cell):
          for j in range(0, num_func):
              for l in range(0,4):
                  matrix[i][1][1][l]=0


        #Greedy Centralized
        for i in range(0,num_cell):
          for j in range(0, num_func):
              for l in range(4,10):
                  matrix[i][0][1][l]=0

        # #Greedy Distributed
        # for i in range(0,num_cell):
        #   for j in range(0, num_func):
        #       for l in range(0,4):
        #           matrix[i][0][1][l]=0

        #edge cloud constraints
        for i in range(1, 4):
            for j in range(0,num_slice):
                for k in range(0, num_func):
                    matrix[i-1][j][k][6]=0
                    matrix[i-1][j][k][7]=0
                    matrix[i-1][j][k][8]=0
                    matrix[i-1][j][k][9]=0
        
        for i in range(4, 7):
            for j in range(0,num_slice):
                for k in range(0, num_func):
                    matrix[i-1][j][k][4]=0
                    matrix[i-1][j][k][5]=0
                    matrix[i-1][j][k][8]=0
                    matrix[i-1][j][k][9]=0
        
        for i in range(7, 10):
            for j in range(0,num_slice):
                for k in range(0, num_func):
                    matrix[i-1][j][k][4]=0
                    matrix[i-1][j][k][5]=0
                    matrix[i-1][j][k][6]=0
                    matrix[i-1][j][k][7]=0
                    
        
        return matrix
        

    def compute_action_mask(self, is_done):
        """
        Compute the set of action masks based on the current state
        """
        action_mask = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        if is_done:
            return action_mask

        # 這次動作的相反動作屬於無效動作
     #   odopa = self.compute_opposite_direction_of_previous_action()
     #   action_mask[odopa] = 0

       
        if self.state[15]==0:
            action_mask[0] = 0
            action_mask[1] =0
            action_mask[2]=0
            action_mask[3]=0

        if self.state[16]== 1 and self.state[15]==1:
            action_mask[0] = 0
            action_mask[1] =0
            action_mask[2]=0
            action_mask[3]=0

        #Greedy Centralization
        if self.state[16]== 0 and self.state[15]==1:
            action_mask[4] = 0
            action_mask[5] =0
            action_mask[6]=0
            action_mask[7]=0
            action_mask[8]=0
            action_mask[9]=0
        
        # #Greedy Distiributed
        # if self.state[16]== 0 and self.state[15]==1:
        #     action_mask[0] = 0
        #     action_mask[1] =0
        #     action_mask[2]=0
        #     action_mask[3]=0

        #Edge cloud Constraint
        if self.state[14]== 1 or self.state[14]== 2 or self.state[14]== 3:
            action_mask[6] = 0
            action_mask[7] = 0
            action_mask[8] = 0
            action_mask[9] = 0

        if self.state[14]== 4 or self.state[14]== 5 or self.state[14]== 6:
            action_mask[4] = 0
            action_mask[5] = 0
            action_mask[8] = 0
            action_mask[9] = 0

        if self.state[14]== 7 or self.state[14]== 8 or self.state[14]== 9:
            action_mask[4] = 0
            action_mask[5] = 0
            action_mask[6] = 0
            action_mask[7] = 0
        
        
        for i in range(0,10):
            if self.state[i]< self.state[17]:
                action_mask[i] = 0 

        count=0
        for i in range(0,10):
            if action_mask[i] == 0:
                count+=1
        if count==10:
            action_mask = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#         if np.count_nonzero(action_mask == 0) == 16:
#             del action_mask
#             action_mask = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1])
#             action_mask[0] = 0

        return action_mask  

env = DummyVecEnv([lambda: REEMAP()])
model = PPO2(MlpPolicy, env, verbose=1)


# In[ ]:


model.learn(total_timesteps=5000000)


# In[ ]:


model.save("REEMAP_model")

reward1=0
reward=0
cost=0
profit=0
cost1=0
profit1=0
obs = env.reset()
states = None
action_masks = []
print('First obs=', obs)
dones=False

while not dones:
    print('action mask is=',action_masks)
    action, _states = model.predict(obs, states,action_mask=action_masks)
    print('action is=', action)
    obs, rewards, dones, infos = env.step(action)
    reward1=reward1+rewards
    action_masks.clear()
    for info in infos:
        env_action_mask = info.get('action_mask')
        action_masks.append(env_action_mask) 
        place=info['place']
        cost1=cost1+info['c']
        cost=info['c']
    print('obs=', obs, 'action=', action)
    print('reward=', rewards, 'done=', dones)
    print('cost is',cost)

#fileptr1.close()
print(place)
print("Total rewards")
print(reward1)
print("Cost1 is")
print(cost1)
