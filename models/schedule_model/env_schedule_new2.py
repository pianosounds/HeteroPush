import gym
from gym import spaces
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

import sys
sys.path.append('/home/3023zyn/zyn/mycodes/heteropush')

from pyparsing import nums

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from gym.utils.env_checker import check_env

'''

新的设计：
1. observation：[[过去N个端侧精度],[过去N个input精度],[过去N个output准确率],[过去N个know准确率]]
    1.1 accuracy profiling：过去的N个数据的精度是基于已经上传的数据进行线性插值而得到
2. reward：论文中设计的，主要是针对单次选择的难度设计。
3. action：调度策略空间为3

'''

class ScheduleEnv(gym.Env):
    '''
    强化学习进行调度策略，调度策略的环境设计
    所有信息都是可以实时得到的
    '''
    def __init__(self,configs):
        self.window_len=configs['window_len']
        self.small_pred_path = configs['small_pred_path']
        self.large_pred_path = configs['large_pred_path']
        self.input_pred_path = configs['up_input_pred_path']
        self.output_pred_path = configs['up_output_pred_path']
        self.know_pred_path = configs['up_know_pred_path']
        self.mode = configs['mode']
        self.sample_rate = configs['sample_rate']

        # 0：选择input
        # 1：选择output
        # 2：选择know
        self.action_space = spaces.Discrete(3)

        # dim0: 端侧准确率向量
        # dim1: input准确率向量
        # dim2: output准确率向量
        # dim3：know准确率向量
        self.observation_space = spaces.Box(low=0,high=1,shape=(4,self.window_len),dtype=np.float32)

        self.init_small_pred = np.load(self.small_pred_path)
        self.init_large_pred = np.load(self.large_pred_path)

        if self.mode=='train':
            # start = random.randint(0,int(len(self.init_large_pred)*0.5))
            start = int(len(self.init_large_pred)*0.7)
            ends = start+int(len(self.init_large_pred)*self.sample_rate)
            print('start:',start)

            # start = int(len(self.init_large_pred)*self.sample_rate)
            # ends = -1
        elif self.mode=='test':
            start = 0
            ends = -1
        else:
            start = 0
            ends = start+int(len(self.init_large_pred)*self.sample_rate)

        self.small_pred = self.init_small_pred[start:ends]
        self.large_pred = self.init_large_pred[start:ends]

        self.input_pred = np.load(self.input_pred_path)
        self.output_pred = np.load(self.output_pred_path)
        self.know_pred = np.load(self.know_pred_path)

        self.frame_nums = len(self.small_pred)

        self.profile_endmodel_accuracy=np.zeros(self.frame_nums)
        self.profile_input_filter_accuracy=np.zeros(self.frame_nums)
        self.profile_output_filter_accuracy=np.zeros(self.frame_nums)
        self.profile_know_filter_accuracy=np.zeros(self.frame_nums)
        self.profile_agent_filtering_accuracy = np.zeros(self.frame_nums)

        print(f'used data nums:{self.frame_nums}')


        self.upload_info = []
        self.cur_index = 0

        self.total_r = 0
        self.done = False
    # ------------------------------observation part--------------------------------
    def profile_current_endmodel_accuracy(self,win_start,win_end):
        '''
        当前的决策已经做了
        输入窗口的起始和终止
        输出该窗口插值的准确率
        '''
        win_small_pred = self.small_pred[win_start:win_end]
        win_large_pred = self.large_pred[win_start:win_end]
        win_upload = self.upload_info[win_start:win_end]
        
        upedx = []
        for i in range(len(win_upload)):
            if win_upload[i]>0:
                upedx.append(i)
        uped_x = np.array(upedx)
        uped_y = np.array([1 if win_small_pred[j]==win_large_pred[j] else 0 for j in uped_x])

        interp_x = []
        for i in range(len(win_upload)):
            if i not in upedx:
                interp_x.append(i)
        interp_x = np.array(interp_x)

        if len(upedx)>0:
            interp_y = np.interp(interp_x,uped_x,uped_y)
        else:
            interp_y = np.array([0])

        profiled_acc = (sum(uped_y)+sum(interp_y))/len(win_upload)
        # print(win_small_pred,win_large_pred,uped_y,interp_y,profiled_acc)
        # print(uped_y,interp_y,profiled_acc)

        return profiled_acc

    def profile_current_filter_accuracy(self,filter,win_start,win_end):
        '''
        当前的决策已经做了
        输入窗口的起始和终止 和 过滤器类型
        输出该窗口的过滤器的准确率预估
        '''
        win_small_pred = self.small_pred[win_start:win_end]
        win_large_pred = self.large_pred[win_start:win_end]
        win_upload = self.upload_info[win_start:win_end]
        if filter=='input':
            win_filter_upload = self.input_pred[win_start:win_end]
        elif filter=='output':
            win_filter_upload = self.output_pred[win_start:win_end]
        elif filter=='know':
            win_filter_upload = self.know_pred[win_start:win_end]
        
        upedx = []
        for i in range(len(win_upload)):
            if win_filter_upload[i]>0:
                upedx.append(i)
        uped_x = np.array(upedx)
        upedy = []
        for i in upedx:
            # 对于已经上传的数据，知道对不对
            if win_filter_upload[i]==1 and win_small_pred[i]!=win_large_pred[i]:
                upedy.append(1)
            elif win_filter_upload[i]==0 and win_small_pred[i]==win_large_pred[i]:
                upedy.append(1)
            else:
                upedy.append(0) 
        uped_y = np.array(upedy)

        interp_x = []
        for i in range(len(win_upload)):
            if i not in upedx:
                interp_x.append(i)
        interp_x = np.array(interp_x)

        if len(upedx)>0:
            # print(uped_x,uped_y)
            interp_y = np.interp(interp_x,uped_x,uped_y)
        else:
            interp_y = np.zeros((win_end-win_start))

        profiled_acc = (sum(uped_y)+sum(interp_y))/len(win_filter_upload)

        return profiled_acc
    
    def calculate_current_system_accuracy(self,win_start,win_end):
        win_small_pred = self.small_pred[win_start:win_end]
        win_large_pred = self.large_pred[win_start:win_end]
        win_upload = self.upload_info[win_start:win_end]

        rightnum = 0
        for i in range(len(win_upload)):
            if win_upload[i]==1:
                rightnum+=1
            elif win_small_pred[i]==win_large_pred[i]:
                rightnum+=1
        win_sys_acc = rightnum/len(win_upload) if len(win_upload)>0 else 0
        # print(len(win_upload))
        return win_sys_acc

    def actual_current_endmodel_accuracy(self,win_start,win_end):
        win_small_pred = self.small_pred[win_start:win_end]
        win_large_pred = self.large_pred[win_start:win_end]
        win_upload = self.upload_info[win_start:win_end]

        rightnum = sum(win_small_pred==win_large_pred)
        win_acc = rightnum/len(win_upload)
        return win_acc
    
    def actual_current_filter_accuracy(self,filter,win_start,win_end):
        win_small_pred = self.small_pred[win_start:win_end]
        win_large_pred = self.large_pred[win_start:win_end]
        win_upload = self.upload_info[win_start:win_end]
        if filter=='input':
            win_filter_upload = self.input_pred[win_start:win_end]
        elif filter=='output':
            win_filter_upload = self.output_pred[win_start:win_end]
        elif filter=='know':
            win_filter_upload = self.know_pred[win_start:win_end]

        filter_rights = 0
        for i in range(len(win_filter_upload)):
            if win_filter_upload[i]==1:
                if win_small_pred[i]!=win_large_pred[i]:
                    filter_rights+=1
            if win_filter_upload[i]==0:
                if win_small_pred[i]==win_large_pred[i]:
                    filter_rights+=1

        act_filter_acc = filter_rights/len(win_upload)
        return act_filter_acc

    def actual_agent_filtring_accuracy(self,win_start,win_end):
        win_small_pred = self.small_pred[win_start:win_end]
        win_large_pred = self.large_pred[win_start:win_end]
        win_upload = self.upload_info[win_start:win_end]
        filter_rights = 0
        upnums = max(1,len(win_upload))
        for i in range(len(win_upload)):
            if win_small_pred[i]==win_large_pred[i] and win_upload[i]==0:
                filter_rights+=1
            elif win_small_pred[i]!=win_large_pred[i] and win_upload[i]==1:
                filter_rights+=1
        agent_filtering_acc = filter_rights/upnums
        return agent_filtering_acc

    def reset(self) :
        '''
        从第0帧开始预测时reset
        '''
        self.upload_info=[]
        self.cur_index=0
        self.done=False
        self.total_r=0

        observation = np.array([np.zeros(self.window_len),np.zeros(self.window_len),np.zeros(self.window_len),np.zeros(self.window_len)],dtype=np.float32)
        return observation
    
    def step(self, action):

        # 手动reset
        if self.done:
            return self.reset()
        
        if self.cur_index==self.frame_nums:
            self.done=True
            # observation = np.array([0.,0.,0.,0.,0.,0.,0.],dtype=np.float32)
            observation = np.array([0.,0.,0.],dtype=np.float32)
            return observation,0,True,{}
        else:
            
            up_input = self.input_pred[self.cur_index]
            up_output = self.output_pred[self.cur_index]
            up_know = self.know_pred[self.cur_index]
            if action==0:
                # 调度input的方法
                cur_up = up_input
            if action==1:
                # 调度output的方法
                cur_up = up_output
            if action==2:
                # 调度know的方法
                cur_up = up_know

            self.upload_info.append(cur_up)

            start = max(0,self.cur_index-self.window_len+1)
            end = self.cur_index+1

            end_model_acc = self.profile_current_endmodel_accuracy(start,end)
            input_filter_acc = self.profile_current_filter_accuracy('input',start,end)
            output_filter_acc = self.profile_current_filter_accuracy('output',start,end)
            know_filter_acc = self.profile_current_filter_accuracy('know',start,end)

            # end_model_acc = self.actual_current_endmodel_accuracy(start,end)
            # input_filter_acc = self.actual_current_filter_accuracy('input',start,end)
            # output_filter_acc = self.actual_current_filter_accuracy('output',start,end)
            # know_filter_acc = self.actual_current_filter_accuracy('know',start,end)
            

            self.profile_endmodel_accuracy[self.cur_index]=end_model_acc
            self.profile_input_filter_accuracy[self.cur_index]=input_filter_acc
            self.profile_output_filter_accuracy[self.cur_index]=output_filter_acc
            self.profile_know_filter_accuracy[self.cur_index]=know_filter_acc

            pad_bits = abs(min(0,self.cur_index-self.window_len+1))
            # print(self.cur_index,pad_bits)
            ends_model_acc_seq = np.pad(self.profile_endmodel_accuracy[start:end],(pad_bits,0),'constant',constant_values=(0,))
            input_filter_acc_seq = np.pad(self.profile_input_filter_accuracy[start:end],(pad_bits,0),'constant',constant_values=(0,))
            output_filter_acc_seq = np.pad(self.profile_output_filter_accuracy[start:end],(pad_bits,0),'constant',constant_values=(0,))
            know_filter_acc_seq = np.pad(self.profile_know_filter_accuracy[start:end],(pad_bits,0),'constant',constant_values=(0,))

            observation = np.array([ends_model_acc_seq,input_filter_acc_seq,output_filter_acc_seq,know_filter_acc_seq],dtype=np.float32)
            
            # 分别计算上传和不上传时候的 代理准确率
            pre_end = max(0,end-1)
            pre_acc = self.actual_agent_filtring_accuracy(start,pre_end)
            # 先记录当前的预测结果
            current_ifup = self.upload_info[-1]
            self.upload_info[-1]=1
            uped_acc = self.actual_agent_filtring_accuracy(start,end)
            self.upload_info[-1]=0
            noup_acc = self.actual_agent_filtring_accuracy(start,end)
            # 将上传结果复位
            self.upload_info[-1]=current_ifup

            uped_improve_acc = uped_acc-pre_acc
            noup_improve_acc = noup_acc-pre_acc

            # 计算reward
            # upload_acc_dict = {0:noup_acc,1:uped_acc}
            upload_acc_dict = {0:noup_improve_acc,1:uped_improve_acc}
            acts_upload_dict = {0:up_input,1:up_output,2:up_know}
            acts_list = [0,1,2]

            
            # 正误奖励
            reward1 = upload_acc_dict[acts_upload_dict[action]]

            # 难度奖励
            reward2 = upload_acc_dict[acts_upload_dict[action]] - sum([upload_acc_dict[acts_upload_dict[i]] for i in acts_list])/len(acts_list)

            # print('accs:',pre_acc,uped_acc,noup_acc,'\t rewards:',reward1,reward2,'\t actions:',action,acts_upload_dict[action],'\t syspreds:',self.small_pred[self.cur_index],self.large_pred[self.cur_index])

            reward = reward1+10*reward2
            # reward = reward1 + 1.2*reward2
            # print(self.cur_index,'\t',up_input,up_output,up_know,'\t',action,'\t')
            self.cur_index+=1
            if self.cur_index==self.frame_nums:
                self.done=True
            self.total_r+=reward

            # print(observation,action,reward,reward1,reward2)

            return observation,reward,self.done,{}

            
def eval_results_torch(env,model):
    # 跑一遍，得到agent的调度策略
    obs = env.reset()
    acts = []
    count = 0
    while True:
        action,_states = model.predict(obs,deterministic=True)
        acts.append(int(action))
        obs,reward,done,info = env.step(action)
        count+=1

        if done:
            obs = env.reset()
            break
    # 拿到全部数据
    small_pred = env.small_pred
    large_pred = env.large_pred

    input_pred = env.input_pred
    output_pred = env.output_pred
    know_pred = env.know_pred

    datanums = len(small_pred)
    print(f'acts_len:{len(acts)}')
    # acts.pop(0)
    print(f'final_acts_len:{len(acts)}')
    # 得到最终的系统预测
    sys_pred = []
    for i in range(datanums):
        act = acts[i]
        if act==0:
            sys_pred.append(input_pred[i])
        elif act==1:
            sys_pred.append(output_pred[i])
        elif act==2:
            sys_pred.append(know_pred[i])
    small_large_equal = small_pred==large_pred
    need_uped = [1 if eq==False else 0 for eq in small_large_equal]
    canbetrue = [1 if need_uped[i] in [input_pred[i],output_pred[i],know_pred[i]] else 0  for i in range(len(small_pred))]
    up_rights = [1 if need_uped[i]==sys_pred[i] else 0 for i in range(len(need_uped))]

    # print(need_uped)
    # print(sys_pred)
    # print(canbetrue)
    print(sum(up_rights),sum(canbetrue))
    # 计算初始系统准确率
    init_sys_right = 0
    for i in range(datanums):
        if small_pred[i] == large_pred[i]:
            init_sys_right+=1
    init_sys_acc = init_sys_right / datanums

    # 计算预测上传的准确率
    right_num = 0
    for i in range(datanums):
        if small_pred[i]==large_pred[i]:
            if sys_pred[i]==0:
                right_num+=1
        else:
            if sys_pred[i]==1:
                right_num+=1
    up_acc = right_num / datanums

    # 计算上传的数据中，该上传的比例
    uped_right = 0
    for i in range(datanums):
        if sys_pred[i]==1:
            if small_pred[i]!=large_pred[i]:
                uped_right+=1
    uped_acc = uped_right/sum(sys_pred) if sum(sys_pred)>0 else 0

    # 计算系统的准确率
    sys_right = 0
    for i in range(datanums):
        if sys_pred[i]==1:
            sys_right+=1
        else:
            if small_pred[i]==large_pred[i]:
                sys_right+=1
    sys_acc = sys_right / datanums

    uprate = sum(sys_pred) / datanums
    gts = [1-int(i) for i in small_pred==large_pred]
    return init_sys_acc,up_acc,uped_acc,sys_acc,uprate,acts,sys_pred,gts

    

small_name = 'yolov3_320'
golden_name = 'yolov5x6_1280'

interval=1
small_pred_path = '/raid/workspace/zhangyinan/s3backup/alphaPrime/uper/experiments/exp_uadetrac/preds_results/labels_preds_mobilenet_'+str(interval)+'.npy'
large_pred_path = '/raid/workspace/zhangyinan/s3backup/alphaPrime/uper/experiments/exp_uadetrac/preds_results/labels_preds_resnet_'+str(interval)+'.npy'

input_upload_path = '../results/input/input_winlen25_upload_results.npy'
# output_upload_path = '../results/output/output_winlen25_upload_results.npy'
output_upload_path = '../results/output/output_winlen25_upload_results.npy'
know_upload_path = '../results/know/know_winlen60_upload_results.npy'

input_preds_path = '/raid/workspace/zhangyinan/s3backup/alphaPrime/uper/experiments/exp_uadetrac/results/input/input_winlen25_upload_results.npy'
output_preds_path = '/raid/workspace/zhangyinan/s3backup/alphaPrime/uper/experiments/exp_uadetrac/results/output/output_winlen25_upload_results.npy'
know_preds_path = '/raid/workspace/zhangyinan/s3backup/alphaPrime/uper/experiments/exp_uadetrac/results/know/know_winlen60_upload_results.npy'



if __name__=='__main__':
    configs = {
        'window_len':30,
        'small_pred_path':small_pred_path,
        'large_pred_path':large_pred_path,
        'up_input_pred_path':input_preds_path,
        'up_output_pred_path':output_preds_path,
        'up_know_pred_path':know_preds_path,
        'sample_rate':0.5,
        'mode':'train'
    }

    schenv = ScheduleEnv(configs)

    time_step = schenv.reset()
    print(time_step)
    total_r = 0
    actions = np.random.randint(low=0,high=3,size=1000)
    print(actions)
    for a in actions:
        time_step = schenv.step(a)
        print(time_step)
        total_r+=time_step[1]
    print("Final reward = ", total_r)