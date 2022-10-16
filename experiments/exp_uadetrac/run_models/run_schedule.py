'''
    调度策略：根据不同的时刻，选择不同的方法，用该方法作为最终的输出
'''
import json
import gym
import sys
import os
sys.path.append("/raid/workspace/zhangyinan/s3backup/alphaPrime/heteropush")
from models.schedule_model.env_schedule_new2 import ScheduleEnv,eval_results_torch

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import numpy as np
import time

os.environ['CUDA_VISIBLE_DEVICES']='0'

exp_name = 'halfreward'

model_path = '../save_weights/scheduleNew/scheduleNewR_dqn_cartpole'

interval = 1
train_step = 130000
sample_rate = 0.5
winlen = 30

def train(configs):
    env = ScheduleEnv(configs)
    model = DQN("MlpPolicy", 
                env, 
                buffer_size=50000,
                batch_size=128,
                learning_rate=0.00001,
                learning_starts=10000,
                # target_update_interval=256,
                policy_kwargs={'net_arch':[256,256]},
                tensorboard_log='./tensorboard/runschedule3',
                verbose=1)
    print('learning.......')
    
    start_time = time.time()

    train_step = configs['train_step']
    model.learn(total_timesteps=train_step,log_interval=5)
    
    end_time = time.time()

    trained_model_path = configs['trained_model_path']
    model.save(trained_model_path)

    print(f'training steps:{train_step} steps, training time:{end_time-start_time}s')

    return model

def test_all():
    '''
    After training all parameters composition
    Test all models
    '''
    interval = 1
    train_step = 80000
    sample_rate = 0.1
    winlen = 30
    for sample_rate in [0.1,0.3,0.5]:
        for winlen in [10,30,60,120]:

            trained_model_path =  model_path +'_trainstep'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)

            acts_schedule_res_path = '../results/scheduleNew2/choose_trainsteps'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)+'.npy'
            upload_schedule_res_path = '../results/scheduleNew2/uploads_trainsteps'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)+'.npy'


            weak_pred_path = '../preds_results/labels_preds_mobilenet_'+str(interval)+'.npy'
            strong_pred_path = '../preds_results/labels_preds_resnet_'+str(interval)+'.npy'

            input_upload_path = '../results/input/input_winlen25_upload_results.npy'
            # output_upload_path = '../results/output/output_winlen25_upload_results.npy'
            output_upload_path = '../results/output/output_winlen25_upload_results.npy'
            know_upload_path = '../results/know/know_winlen60_upload_results.npy'

            configs = {
                'train_step':train_step,
                'window_len':winlen,
                'small_pred_path':weak_pred_path,
                'large_pred_path':strong_pred_path,
                'up_input_pred_path':input_upload_path,
                'up_output_pred_path':output_upload_path,
                'up_know_pred_path':know_upload_path,
                'trained_model_path':trained_model_path,
                'sample_rate':sample_rate,
                'mode':'test',
            }

            env = ScheduleEnv(configs)

            print('load and testing model...')
            model = DQN.load(trained_model_path)

            start_time = time.time()
            init_sys_acc,upacc,uped_acc,sys_acc,uprate,acts,sys_pred,gts = eval_results_torch(env,model)
            end_time = time.time()
            print('inference time:',end_time-start_time)

            print(f'init_sys_acc:{init_sys_acc} upacc:{upacc} upedacc:{uped_acc} sys_acc:{sys_acc} uprate:{uprate}')
            
            acts = np.array(acts,dtype=np.int32)
            sys_pred = np.array(sys_pred,dtype=np.int32)
            np.save(acts_schedule_res_path,acts)
            np.save(upload_schedule_res_path,sys_pred)

    return init_sys_acc,upacc,uped_acc,sys_acc,uprate,acts,sys_pred,gts

def test_single():

    trained_model_path =  model_path +'_trainstep'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)

    acts_schedule_res_path = '../results/scheduleNew/choose_init_balance'+'_trainsteps'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)+'.npy'
    upload_schedule_res_path = '../results/scheduleNew/uploads_init_balance'+'_trainsteps'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)+'.npy'

    weak_pred_path = '../preds_results/labels_preds_mobilenet_'+str(interval)+'.npy'
    strong_pred_path = '../preds_results/labels_preds_resnet_'+str(interval)+'.npy'

    input_upload_path = '../results/input/input_winlen25_upload_results.npy'
    output_upload_path = '../results/output/output_winlen25_upload_results.npy'
    know_upload_path = '../results/know/know_winlen60_upload_results.npy'
    configs = {
        'train_step':train_step,
        'window_len':winlen,
        'small_pred_path':weak_pred_path,
        'large_pred_path':strong_pred_path,
        'up_input_pred_path':input_upload_path,
        'up_output_pred_path':output_upload_path,
        'up_know_pred_path':know_upload_path,
        'trained_model_path':trained_model_path,
        'sample_rate':sample_rate,
        'mode':'test',
    }

    env = ScheduleEnv(configs)

    print('load and testing model...')
    model = DQN.load(trained_model_path)

    start_time = time.time()
    init_sys_acc,upacc,uped_acc,sys_acc,uprate,acts,sys_pred,gts = eval_results_torch(env,model)
    end_time = time.time()
    print('inference time:',end_time-start_time)

    print(f'init_sys_acc:{init_sys_acc} upacc:{upacc} upedacc:{uped_acc} sys_acc:{sys_acc} uprate:{uprate}')
    
    acts = np.array(acts,dtype=np.int32)
    sys_pred = np.array(sys_pred,dtype=np.int32)
    np.save(acts_schedule_res_path,acts)
    np.save(upload_schedule_res_path,sys_pred)

    return init_sys_acc,upacc,uped_acc,sys_acc,uprate,acts,sys_pred,gts


def train_all():

    interval = 1
    train_step = 130000
    sample_rate = 0.1
    winlen = 120
    for train_step in [80000,130000]:
        for sample_rate in [0.1,0.3,0.5]:
            for winlen in [10,30,60,120]:

                trained_model_path =  model_path +'_trainstep'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)

                acts_schedule_res_path = '../results/scheduleNew2/choose_trainsteps'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)+'.npy'
                upload_schedule_res_path = '../results/scheduleNew2/uploads_trainsteps'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)+'.npy'


                weak_pred_path = '../preds_results/labels_preds_mobilenet_'+str(interval)+'.npy'
                strong_pred_path = '../preds_results/labels_preds_resnet_'+str(interval)+'.npy'

                input_upload_path = '../results/input/input_winlen25_upload_results.npy'
                # output_upload_path = '../results/output/output_winlen25_upload_results.npy'
                output_upload_path = '../results/output/output_winlen25_upload_results.npy'
                know_upload_path = '../results/know/know_winlen60_upload_results.npy'

                configs = {
                    'train_step':train_step,
                    'window_len':winlen,
                    'small_pred_path':weak_pred_path,
                    'large_pred_path':strong_pred_path,
                    'up_input_pred_path':input_upload_path,
                    'up_output_pred_path':output_upload_path,
                    'up_know_pred_path':know_upload_path,
                    'trained_model_path':trained_model_path,
                    'sample_rate':sample_rate,
                    'mode':'train',
                }
                print(f'------trainstep:',train_step,' trainrate:',sample_rate,' winlen:',winlen,'---------')
                train(configs)

def train_single():

    trained_model_path =  model_path +'_trainstep'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)

    acts_schedule_res_path = '../results/scheduleNewSens/choose_trainsteps'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)+'.npy'
    upload_schedule_res_path = '../results/scheduleNewSens/uploads_trainsteps'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)+'.npy'


    weak_pred_path = '../preds_results/labels_preds_mobilenet_'+str(interval)+'.npy'
    strong_pred_path = '../preds_results/labels_preds_resnet_'+str(interval)+'.npy'

    input_upload_path = '../results/input/input_winlen25_upload_results.npy'
    # output_upload_path = '../results/output/output_winlen25_upload_results.npy'
    output_upload_path = '../results/output/output_winlen25_upload_results.npy'
    know_upload_path = '../results/know/know_winlen60_upload_results.npy'

    configs = {
        'train_step':train_step,
        'window_len':winlen,
        'small_pred_path':weak_pred_path,
        'large_pred_path':strong_pred_path,
        'up_input_pred_path':input_upload_path,
        'up_output_pred_path':output_upload_path,
        'up_know_pred_path':know_upload_path,
        'trained_model_path':trained_model_path,
        'sample_rate':sample_rate,
        'mode':'train',
    }
    print(f'------trainstep:',train_step,' trainrate:',sample_rate,' winlen:',winlen,'---------')
    train(configs)



if __name__=='__main__':
    train_single()
    test_single()