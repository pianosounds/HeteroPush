import json
import gym
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import numpy as np
import time

sys.path.append("/raid/workspace/zhangyinan/s3backup/alphaPrime/heteropush")

from models.dqn_model.env_profile import ACTEnv,eval_results_torch

train_step = 140000

model_path2 = '../save_weights/dqn_forward/dqn_cartpole_explore_forward'+'_'+str(train_step)+'_8000'
res_path = '../results/dqn_forward/uploads_dqn_forward_'+str(train_step)+'_8000_'+'.npy'

small_name = 'yolov3_320'
golden_name = 'yolov5x6_1280'

small_pred_path = '../preds_results/labels_preds_mobilenet_1.npy'
large_pred_path = '../preds_results/labels_preds_resnet_1.npy'

configs = {
        'window_len':90,
        'small_pred_path':small_pred_path,
        'large_pred_path':large_pred_path,
        'mode':'test'
    }


def main():

    env = ACTEnv(configs)
    model = DQN("MlpPolicy",
                env, 
                buffer_size=50000,
                batch_size=128,
                learning_starts=10000,
                verbose=1)
    '''
    After experiments, different step with different training times as follows:
    100step 3s
    200step 9s
    300step 13s
    400step 18s
    1000step 48s
    10000step 530s
    100000step 1770s
    300000step 1770s
    500000step 5780s
    '''
    mode = configs['mode']
    if mode=='train':
        print('learning....')
        start_time = time.time()
        model.learn(total_timesteps=train_step, log_interval=1)
        print('learn finished,! saved!')
        end_time = time.time()
        print(f'training steps:{train_step} steps, training time:{end_time-start_time}s')
        model.save(model_path2)

    del model # remove to demonstrate saving and loading
    print('load and test model...')
    model = DQN.load(model_path2)

    start_time = time.time()
    init_sys_acc,upacc,uped_acc,sys_acc,uprate,acts,gts = eval_results_torch(env,model)
    end_time = time.time()

    print(f'init_sys_acc:{init_sys_acc} upacc:{upacc} upedacc:{uped_acc} sys_acc:{sys_acc} uprate:{uprate} testing_time:{end_time-start_time}')
    acts = np.array(acts)
    np.save(res_path,acts)

if __name__=='__main__':
    main()