import json
import gym
import sys
from env_profile import ACTEnv,eval_results_torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import numpy as np
import time

sys.path.append("/home/3023zyn/zyn/mycodes/mobileDydamicUpload")

frame_dir = '/raid/workspace/zyn/Datasets/Deqing/54/'
# env = gym.make("CartPole-v0")
model_path = './weights/dqn_cartpole'


train_step = 160000
model_path2 = './weights/dqn_cartpole2_explore'+'_'+str(train_step)+'_8000'
model_path2 = './weights/dqn_cartpole2_explore_v2'+'_'+str(train_step)+'_8000'
res_path = '/home/3023zyn/zyn/mycodes/uper/results/dqn_preds/actions_v2_'+str(train_step)+'_8000_'+'.json'

small_name = 'yolov3_320'
# small_name = 'yolov3-spp_608'
golden_name = 'yolov5x6_1280'

small_pred_path = '/raid/workspace/zyn/videoCEedgecloud/city/count_res/54_'+small_name+'.npy'
large_pred_path = '/raid/workspace/zyn/videoCEedgecloud/city/count_res/54_'+golden_name+'.npy'

configs = {
        'window_len':30,
        'small_pred_path':small_pred_path,
        'large_pred_path':large_pred_path
    }


def main():
    

    # print(small_pred==large_pred)

    env = ACTEnv(configs)
    # env = DummyVecEnv([lambda : env])
    model = DQN("MlpPolicy", 
                env, 
                buffer_size=50000,
                batch_size=128,
                verbose=1)
    '''
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

    train=0
    if train==1:
        print('learning....')
        start_time = time.time()
        model.learn(total_timesteps=train_step, log_interval=3)
        print('learn finished,! saved!')
        end_time = time.time()

        print(f'training steps:{train_step} steps, training time:{end_time-start_time}s')

        model.save(model_path2)

    del model # remove to demonstrate saving and loading
    print('load and test model...')
    model = DQN.load(model_path2)
    
    # obs = env.reset()
    # acts = []
    # count = 0
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     acts.append(int(action))
    #     obs, reward, done, info = env.step(action)
    #     count+=1  
    #     # env.render()
    #     if count%100==0:
    #         print(f'--------------------------{sum(acts)}-----------------------')
    #         print(obs,reward,done,info,count)
    #     if done:
    #         obs = env.reset()
    #         break

    init_sys_acc,upacc,uped_acc,sys_acc,uprate,acts,gts = eval_results_torch(env,model)

    print(f'init_sys_acc:{init_sys_acc} upacc:{upacc} upedacc:{uped_acc} sys_acc:{sys_acc} uprate:{uprate}')
    # print(sum(small_pred==large_pred))
    with open(res_path,'w') as f:
        f.write(json.dumps(acts))

if __name__=='__main__':
    main()