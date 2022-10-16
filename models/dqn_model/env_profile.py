import gym
from gym import spaces
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import cv2
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from gym.utils.env_checker import check_env


class ACTEnv(gym.Env):
    '''
    Acc Communication Trade Off Enviroment

    simulate environment
    '''
    def __init__(self,configs):
        self.window_len = configs['window_len']
        self.small_pred_path = configs['small_pred_path']
        self.large_pred_path = configs['large_pred_path']
        
        
        # 0: not upload
        # 1: upload
        self.action_space = spaces.Discrete(2)

        # Note:

        # dim1: accuracy of current inference model
        # dim2: upload rate of current inference model

        # dim3: accuracy of current system
        # dim3: upload rate of current system

        # dim4: accuracy of current frame uploaded
        # dim4: upload rate of current frame uploaded

        # dim5: accuracy of current frame not uploaded
        # dim5: upload rate of current frame not uploaded
        


        self.observation_space = spaces.Box(low=-1,high=1,shape=(5,),dtype=np.float32)

        # observation create
        start = 0
        nums = -1
        self.small_pred = np.load(self.small_pred_path)[start:start+nums]
        self.large_pred = np.load(self.large_pred_path)[start:start+nums]

        self.frame_nums = len(self.small_pred)

        self.upload_info = []
        self.cur_index = 0

        self.total_r = 0

        self.done = False
    
    def calculate_endmodel_accuracy_profile(self):
        '''
        dim1:  accuracy of current model——simulate
        '''
        left = max(0,self.cur_index-self.window_len)
        small_right = 0
        valid_num = 0
        for i in range(left,self.cur_index):
            if self.upload_info[i]==1:
                valid_num+=1
                if self.small_pred[i] == self.large_pred[i]:
                    small_right+=1
        if valid_num>0:
            small_acc = small_right/valid_num
        else:
            small_acc=0.
        return small_acc

    def calculate_endmodel_accuracy(self):
        '''
        dim1:  accuracy of current model——groundtruth
        '''
        left = max(0,self.cur_index-self.window_len)
        small_right = 0
        valid_num = 0
        for i in range(left,self.cur_index):
            valid_num+=1
            if self.small_pred[i] == self.large_pred[i]:
                small_right+=1
        if valid_num>0:
            small_acc = small_right/valid_num
        else:
            small_acc=0.
        return small_acc
    
    def calculate_upload_rate(self):
        '''
        dim2: upload rate of current system
        '''
        left = max(0,self.cur_index-self.window_len)
        upload_window = self.upload_info[left:self.cur_index]
        if len(upload_window)>0:
            upload_rate = sum(upload_window)/len(upload_window)
        else:
            upload_rate = 0.
        return upload_rate

    def calculate_upload_acc(self):
        '''
        dim3: accuracy of the upload frames
        '''
        uped_right = 0
        uped_num = 0
        left = max(0,self.cur_index-self.window_len)
        upload_window = self.upload_info[left:self.cur_index]
        for i in range(left,self.cur_index):
            if self.upload_info[i]==1:
                uped_num+=1
                if self.small_pred[i] != self.large_pred[i]:
                    uped_right+=1
        if uped_num>0:
            uped_acc = uped_right/uped_num
        else:
            uped_acc=0.
        return uped_acc

    def calculate_system_accuracy(self):
        '''
        accuracy of current system
        '''
        left = max(0,self.cur_index-self.window_len)
        sys_right = 0
        valid_num = 0
        for i in range(left,self.cur_index):
            if self.upload_info[i]==1:
                sys_right+=1
            else:
                if self.small_pred[i]==self.large_pred[i]:
                    sys_right+=1
        if self.cur_index>left:
            sys_acc = sys_right / (self.cur_index-left)
        else:
            sys_acc = 0.
        return sys_acc
    
    def simulate_upload_accuracy(self):
        '''
        simulate the accuracy of DQN if the current frame been uploaded
        '''
        left = max(0,self.cur_index-self.window_len)
        up_right = 0
        for i in range(left,self.cur_index):
            if self.upload_info[i]==1:
                if self.small_pred[i]!=self.large_pred[i]:
                    up_right+=1
            else:
                if self.small_pred[i]==self.large_pred[i]:
                    up_right+=1
        if self.small_pred[self.cur_index]!=self.large_pred[self.cur_index]:
            up_right+=1
        up_lens = self.cur_index-left+1
        up_acc = up_right / up_lens
        return up_acc
    
    def simulate_noupload_accuracy(self):
        '''
        simulate the accuracy of DQN if the current frame not been uploaded
        '''
        left = max(0,self.cur_index-self.window_len)
        up_right = 0
        for i in range(left,self.cur_index):
            if self.upload_info[i]==1:
                if self.small_pred[i]!=self.large_pred[i]:
                    up_right+=1
            else:
                if self.small_pred[i]==self.large_pred[i]:
                    up_right+=1
        if self.small_pred[self.cur_index]==self.large_pred[self.cur_index]:
            up_right+=1
        up_lens = self.cur_index-left+1
        up_acc = up_right / up_lens
        return up_acc

    def simulate_upload_sys_accuracy(self):
        '''
        simulate the accuracy of current system if current frame been uploaded
        '''
        left = max(0,self.cur_index-self.window_len)
        sys_right = 0
        for i in range(left,self.cur_index):
            if self.upload_info[i]==1:
                sys_right+=1
            else:
                if self.small_pred[i]==self.large_pred[i]:
                    sys_right+=1
        sys_right+=1
        sys_lens = self.cur_index-left+1
        sys_acc = sys_right / sys_lens
        return sys_acc
    
    def simulate_noupload_sys_accuracy(self):
        '''
        simulate the accuracy of current system if current frame not been uploaded
        '''
        left = max(0,self.cur_index-self.window_len)
        sys_right = 0
        for i in range(left,self.cur_index):
            if self.upload_info[i]==1:
                sys_right+=1
            else:
                if self.small_pred[i]==self.large_pred[i]:
                    sys_right+=1
        if self.small_pred[self.cur_index]==self.large_pred[self.cur_index]:
            sys_right+=1
        sys_lens = self.cur_index-left+1
        sys_acc = sys_right / sys_lens
        return sys_acc
    
    def calculate_total_upload_accuracy(self):
        '''
        caculate the accuracy of current policy if current frame been uploaded
        '''
        correct = 0
        for i in range(len(self.upload_info)):
            if self.small_pred[i]==self.large_pred[i]:
                if self.upload_info[i]==0:
                    correct+=1
            else:
                if self.upload_info[i]==1:
                    correct+=1
        upload_acc = correct/len(self.upload_info)
        return upload_acc
    
    def calculate_total_sys_accuracy(self):
        '''
        caculate the accuracy of current system with the current policy
        '''
        correct = 0
        for i in range(len(self.upload_info)):
            if self.upload_info[i]==1:
                correct+=1
            else:
                if self.small_pred[i]==self.large_pred[i]:
                    correct+=1
        sys_acc = correct/len(self.upload_info)
        return sys_acc

    def reset(self) :
        '''
        reset from the first frame
        '''
        self.upload_info=[]
        self.cur_index=0
        self.done=False
        self.total_r=0

        observation = np.array([0.,0.,0.,0.,0.],dtype=np.float32)
        return observation

    def step(self, action):
        if self.done:
            return self.reset()
        
        if self.cur_index == self.frame_nums:
            self.done=True
            observation = np.array([0.,0.,0.,0.,0.],dtype=np.float32)
            return observation,0,True,{}
        else:
            reward = 0.

            up_sys_acc = self.simulate_upload_sys_accuracy()
            up_acc = self.simulate_upload_accuracy()

            noup_sys_acc = self.simulate_noupload_sys_accuracy()
            noup_acc = self.simulate_noupload_accuracy()
            
            if action==0:
                self.upload_info.append(0)
                lens = min(self.cur_index,self.window_len)
                reward = (noup_sys_acc-up_sys_acc + noup_acc-up_acc) * lens / self.window_len
            if action==1:
                self.upload_info.append(1)
                lens = min(self.cur_index,self.window_len)
                reward = (up_sys_acc-up_sys_acc + up_acc - noup_acc) * lens / self.window_len

            small_acc = self.calculate_endmodel_accuracy_profile()
            upload_rate = self.calculate_upload_rate()
            sys_acc = self.calculate_system_accuracy()
            upload_sys_acc = self.simulate_upload_sys_accuracy()
            noupload_sys_acc = self.simulate_noupload_sys_accuracy()
            observation = np.array([small_acc,upload_rate,sys_acc,upload_sys_acc,noupload_sys_acc],dtype=np.float32)
            self.cur_index+=1
            self.total_r+=reward
            return observation,reward,self.done,{}

def eval_results(env,policy):
    time_step = env.reset()
    while not time_step.is_last():
        act = policy.action(time_step)
        time_step = env.step(act)
    
    actenv = env._env._envs[0]

    upload_acc = actenv.calculate_total_upload_accuracy()
    sys_acc = actenv.calculate_total_sys_accuracy()
    total_r = actenv.total_r

    upload_list =  actenv.upload_info

    return upload_acc,sys_acc,upload_list,total_r

def eval_results_torch(env,model):
    obs = env.reset()
    acts = []
    count = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        acts.append(int(action))
        obs, reward, done, info = env.step(action)
        count+=1  
        if done:
            obs = env.reset()
            break
    
    small_pred = env.small_pred
    large_pred = env.large_pred

    init_sys_right = 0
    for i in range(len(small_pred)):
        if small_pred[i]==large_pred[i]:
            init_sys_right+=1
    init_sys_acc = init_sys_right/len(small_pred)

    right_num=0
    for i in range(len(small_pred)):
        if small_pred[i]==large_pred[i]:
            if acts[i]==0:
                right_num+=1
        else:
            if acts[i]==1:
                right_num+=1
    sys_right = 0
    for i in range(len(small_pred)):
        if acts[i]==1:
            sys_right+=1
        else:
            if small_pred[i]==large_pred[i]:
                sys_right+=1
    sys_acc= sys_right/len(small_pred)

    uped_right = 0
    for i in range(len(acts)):
        if acts[i]==1:
            if small_pred[i]!=large_pred[i]:
                uped_right+=1
    uped_acc = uped_right/sum(acts) if sum(acts)>0 else 0

    print(len(small_pred)==len(acts))
    upacc = right_num/len(acts)
    uprate = sum(acts)/len(acts)
    gts = [1-int(i) for i in small_pred==large_pred]
    return init_sys_acc,upacc,uped_acc,sys_acc,uprate,acts,gts


small_name = 'yolov3_320'
golden_name = 'yolov5x6_1280'

small_pred_path = '/raid/workspace/zyn/videoCEedgecloud/city/count_res/54_'+small_name+'.npy'
large_pred_path = '/raid/workspace/zyn/videoCEedgecloud/city/count_res/54_'+golden_name+'.npy'

if __name__=='__main__':
    configs = {
        'window_len':10,
        'small_pred_path':small_pred_path,
        'large_pred_path':large_pred_path
    }
    actenv = ACTEnv(configs)
    
    time_step = actenv.reset()
    print(time_step)
    total_r = time_step.reward

    actions = np.random.randint(low=0,high=2,size=50)
    for a in actions:
        time_step = actenv.step(a)
        print(time_step)
        total_r+=time_step.reward
    print("Final reward = ", total_r)
