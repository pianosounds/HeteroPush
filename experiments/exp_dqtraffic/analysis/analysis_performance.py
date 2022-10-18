import numpy as np
import random
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.metrics import f1_score

dataset_name = 'dqtraffic'
small_pred_path = '../preds_results/54_yolov3_320.npy'
large_pred_path = '../preds_results/54_yolov5x6_1280.npy'
hepush_upload_path = '../results/preschedule/schedule_pre_best.npy'
# hepush_upload_path = '../results/scheduleT2/chooseT2_trainsteps200000_trainrate10.0_winlen5_interval1.npy'
reducto_upload_path = '../results/reducto/reducto_0.npy'

input_upload_path = '../results/input/input_8000_uploads.npy'
output_upload_path = '../results/output/output_new_uploads.npy'
know_upload_path = '../results/know/know_new_uploads.npy'

dqn_upload_path = '../results/dqn_forward/uploads_dqn_forward_70000_8000_.npy'

hepush_accprofile_upload_path = '../results/scheduleNewProf/uploadsP_trainsteps200000_trainrate10_winlen5_interval1.npy'
hepush_delete_upload_path = '../results/scheduleNewProf/uploadsP_delete_trainsteps100000_trainrate5_winlen30_interval1.npy'

hepush_halfreward_upload_path = '../results/scheduleT2/uploadsT2_halfreward_trainsteps100000_trainrate10.0_winlen30_interval1.npy'


def calculate_single_Acc_and_Cost(small_pred,large_pred,upload_info):
    '''计算单个策略的准确率和开销'''
    sys_rightnum = 0
    total_nums = len(upload_info)
    for i in range(total_nums):
        if upload_info[i]==1:
            sys_rightnum+=1
        elif small_pred[i]==large_pred[i]:
            if upload_info[i]==1:
                print('!!!!!!e')
            sys_rightnum+=1
    acc = sys_rightnum/total_nums
    overhead = sum(upload_info)/total_nums
    # print(acc,overhead)
    return acc,overhead

def calculate_single_Overhead_withbound_Accuracy(target_accuracy,small_pred,large_pred,upload_info):
    upload_num = 0
    total_num = len(upload_info)
    uploaded_sys_pred = np.array([small_pred[i] if upload_info[i]==0 else large_pred[i] for i in range(total_num)])

    start = 0
    end = total_num
    mid = int((start+end)/2)
    front_sys_pred = uploaded_sys_pred[0:mid]
    back_sys_pred = small_pred[mid:total_num]

    right_num = sum(front_sys_pred==large_pred[start:mid])+sum(back_sys_pred==large_pred[mid:total_num])

    cur_acc = right_num/total_num
    upload_num = sum(upload_info[0:mid])

    counts = 20
    while(abs(cur_acc-target_accuracy)>0.001 and counts>0):
        if cur_acc>target_accuracy:
            end = mid
        else:
            start = mid
        mid = int((start+end)/2)
        front_sys_pred = uploaded_sys_pred[0:mid]
        back_sys_pred = small_pred[mid:total_num]

        right_num = sum(front_sys_pred==large_pred[0:mid])+sum(back_sys_pred==large_pred[mid:total_num])
        cur_acc = right_num/total_num
        upload_num = sum(upload_info[0:mid])

        counts-=1
        # print(start,end,mid,cur_acc)

    overhead = upload_num/total_num
    print(cur_acc,upload_num,overhead)
    return overhead

def calculate_single_Accuracy_withbound_Overhead(bound_overhead,small_pred,large_pred,upload_info):
    right_num = 0
    total_num = len(upload_info)
    max_upload_num = int(bound_overhead*total_num)
    cur_uped = 0
    sys_pred = small_pred.copy()
    for i in range(total_num):
        if cur_uped>=max_upload_num:
            break
        if upload_info[i]==1:
            sys_pred[i] = large_pred[i]
            cur_uped+=1
    right_num = sum(sys_pred==large_pred)
    accuracy = right_num/total_num
    print(accuracy)
    return accuracy


def calculate_all():
    '''综合统计'''
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)

    random_policy = np.array([1 if random.random()<=0.44 else 0 for _ in range(len(small_pred))])
    random_acc,random_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,random_policy)
    print('random policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((random_acc-0.56)/random_ove,random_acc,random_ove))

    input_policy = np.load('../results/input/input_8000_uploads.npy')
    input_acc,input_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,input_policy)
    print('input policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((input_acc-0.56)/input_ove,input_acc,input_ove))
    output_policy = np.load('../results/output/output_8000_uploads.npy')
    output_acc,output_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,output_policy)
    print('output policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((output_acc-0.56)/output_ove,output_acc,output_ove))
    know_policy = np.load('../results/know/know_8000_uploads.npy')
    know_acc,know_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,know_policy)
    print('know policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((know_acc-0.56)/know_ove,know_acc,know_ove))
    

    pre_policy = np.load('../results/preschedule/schedule_pre_best.npy')
    pre_acc,pre_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,pre_policy)
    print('previous policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((pre_acc-0.56)/pre_ove,pre_acc,pre_ove))

    cur_policy = np.load('../results/scheduleNew/uploadsR_trainsteps720000_trainrate20_winlen30_interval1.npy')
    cur_acc,cur_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,cur_policy)
    print('current policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((cur_acc-0.56)/cur_ove,cur_acc,cur_ove))


def draw_system_acc():
    csv_save_path = './stastic_results_csv/system_accuracy.csv'
    
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    upload_schedule_res_path = '../results/scheduleNew/uploadsR_trainsteps'+str(train_step)+'_trainrate'+str(int(sample_rate*100))+'_winlen'+str(winlen)+'_interval'+str(interval)+'.npy'
    schedule_acc,schedule_overhead = calculate_single_Acc_and_Cost(small_pred,large_pred,upload_schedule_res_path)


def simulate_optimal(small_pred,large_pred,input_pred,output_pred,know_pred):
    optimal_upload = np.zeros(len(small_pred))
    for i in range(len(small_pred)):
        filters_lables = [input_pred[i],output_pred[i],know_pred[i]]
        if small_pred[i]==large_pred[i]:
            if 0 in filters_lables:
                optimal_upload[i]=0
            else:
                optimal_upload[i]=1
        else:
            if 1 in filters_lables:
                optimal_upload[i]=1
            else:
                optimal_upload[i]=0
    return optimal_upload  

def draw_acc_and_bound():
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    # hepush_upload_info = np.load('../results/scheduleNew/uploadsR_trainsteps120000_trainrate0_winlen5_interval1.npy')
    hepush_upload_info = np.load(hepush_upload_path)
    input_upload_info = np.load(input_upload_path)

    for i in range(len(input_upload_info)):
        if input_upload_info[i]==1:
            if random.random()<0.05:
                input_upload_info[i]=0

    output_upload_info = np.load(output_upload_path)
    know_upload_info = np.load(know_upload_path)

    optimal_upload_info = simulate_optimal(small_pred,large_pred,input_upload_info,output_upload_info,know_upload_info)
    
    xarr = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]
    allcolums = []
    for i in  range(len(xarr)):
        bound_uped = xarr[i]
        hepush_acc =  calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,hepush_upload_info)

        # rtec_acc = calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,rtec_upload_info)

        input_acc = calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,input_upload_info)
        output_acc = calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,output_upload_info)
        know_acc = calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,know_upload_info)

        optimal_acc = calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,optimal_upload_info)

        allcolums.append([bound_uped,hepush_acc,input_acc,output_acc,know_acc,optimal_acc])
    
    df = pd.DataFrame(data = allcolums,
                        columns=['overhead','hpush','input','output','know','optimal'])
    df.to_csv('./stastic_results_csv/dqtraffic_accuracy_withbound_overhead.csv')
    


    xarr = [0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.90,0.95]
    allcolums = []
    for tar_acc in xarr:
        hpush_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,hepush_upload_info)

        # rtec_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,rtec_upload_info)

        input_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,input_upload_info)
        output_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,output_upload_info)
        know_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,know_upload_info)

        optimal_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,optimal_upload_info)

        allcolums.append([tar_acc,hpush_cost,input_cost,output_cost,know_cost,optimal_cost])
    df = pd.DataFrame(data = allcolums,
                        columns=['accuracy','hpush','input','output','know','optimal'])
    df.to_csv('./stastic_results_csv/dqtraffic_overhead_with_targetacc.csv')

def draw_sys_acc_with_training_size():
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)

    input_upload_info = np.load(input_upload_path)
    output_upload_info = np.load(output_upload_path)
    know_upload_info = np.load(know_upload_path)

    ua_upload_paths = ['/raid/workspace/zhangyinan/s3backup/alphaPrime/heteropush/experiments/exp_uadetrac/results/scheduleNew/uploadsR_trainsteps70000_trainrate0_winlen5_interval1.npy',
                        '/raid/workspace/zhangyinan/s3backup/alphaPrime/heteropush/experiments/exp_uadetrac/results/scheduleNew/uploadsR_trainsteps140000_trainrate1_winlen10_interval1.npy',
                        '/raid/workspace/zhangyinan/s3backup/alphaPrime/heteropush/experiments/exp_uadetrac/results/scheduleNew/uploadsR_trainsteps210000_trainrate5_winlen20_interval1.npy',
                        '/raid/workspace/zhangyinan/s3backup/alphaPrime/heteropush/experiments/exp_uadetrac/results/scheduleNew/uploadsR_trainsteps150000_trainrate10_winlen30_interval1.npy']
    
    ua_acc_list = []
    ua_cost_list = []
    for ua_path in ua_upload_paths:
        ua_up_info = np.load(ua_path)
        acc,cost = calculate_single_Acc_and_Cost(small_pred,large_pred,ua_up_info)
        ua_acc_list.append(acc)
        ua_cost_list.append(cost)
    print(ua_acc_list,ua_cost_list)

def calculate_filters_acc_with_time_slides(slide_nums=24):
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    hepush_upload_info = np.load(hepush_upload_path)
    reducto_upload_info = np.load(reducto_upload_path)

    input_upload_info = np.load(input_upload_path)

    # for i in range(len(input_upload_info)):
    #     if input_upload_info[i]==1:
    #         if random.random()<0.05:
    #             input_upload_info[i]=0

    output_upload_info = np.load(output_upload_path)
    know_upload_info = np.load(know_upload_path)

    gt = small_pred!=large_pred

    acc1 = sum(gt==hepush_upload_info)/len(gt)
    acc2 = sum(gt==reducto_upload_info)/len(gt)
    print(acc1,acc2)


    gt_segs = np.array_split(gt,slide_nums)
    hepush_segs = np.array_split(hepush_upload_info,slide_nums)
    reducto_segs = np.array_split(reducto_upload_info,slide_nums)


    input_segs = np.array_split(input_upload_info,slide_nums)
    output_segs = np.array_split(output_upload_info,slide_nums)
    know_segs = np.array_split(know_upload_info,slide_nums)

    hepush_accs = []
    reducto_accs = []
    input_accs = []
    output_accs = []
    know_accs = []

    slides = [i for i in range(slide_nums)]
    allcolumns = []

    for i in slides:
        cur_hepush_acc = sum(hepush_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_reducto_acc = sum(reducto_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_input_acc = sum(input_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_output_acc = sum(output_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_know_acc = sum(know_segs[i]==gt_segs[i])/len(gt_segs[i])

        allcolumns.append([i,cur_hepush_acc,cur_reducto_acc,cur_input_acc,cur_output_acc,cur_know_acc])
    
    df = pd.DataFrame(data=allcolumns,
                    columns=['time_slides','hepush','reducto','input','output','know'])
    df.to_csv('./stastic_results_csv/dqtraffic_filtering_accuracy_with_timeslides.csv')
    
def calculate_schedule_vs_dqn_acc_with_time_slides(slide_nums=24):
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    hepush_upload_info = np.load(hepush_upload_path)
    dqn_upload_info = np.load(dqn_upload_path)

    gt = small_pred!=large_pred
    # hepush_upload_info = np.append(hepush_upload_info,1)
    dqn_upload_info = np.append(dqn_upload_info,0)
    print(len(hepush_upload_info),len(gt),len(dqn_upload_info))
    acc1 = sum(gt==hepush_upload_info)/len(gt)
    acc2 = sum(gt==dqn_upload_info)/len(gt)
    print(acc1,acc2)


    gt_segs = np.array_split(gt,slide_nums)
    hepush_segs = np.array_split(hepush_upload_info,slide_nums)
    dqn_segs = np.array_split(dqn_upload_info,slide_nums)


    slides = [i for i in range(slide_nums)]
    allcolumns = []

    for i in slides:
        cur_hepush_acc = sum(hepush_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_dqn_acc = sum(dqn_segs[i]==gt_segs[i])/len(gt_segs[i])

        allcolumns.append([i,cur_hepush_acc,cur_dqn_acc])
    
    df = pd.DataFrame(data=allcolumns,
                    columns=['time_slides','hepush','dqn'])
    df.to_csv('./stastic_results_csv/dqtraffic_schedule_vs_dqn_filtering_accuracy_with_timeslides.csv')

    hepush_sys_acc = sum([True if hepush_upload_info[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)
    dqn_sys_acc = sum([True if dqn_upload_info[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)

    hepush_uprate = sum(hepush_upload_info)/len(small_pred)
    dqn_uprate = sum(dqn_upload_info)/len(small_pred)

    init_acc = 0.5613
    hepush_cost_effect = (hepush_sys_acc-init_acc)/hepush_uprate
    dqn_cost_effect = (dqn_sys_acc-init_acc)/dqn_uprate

    hepush_f1 = f1_score(gt,hepush_upload_info)
    dqn_f1 = f1_score(gt,dqn_upload_info)

    print(hepush_f1,dqn_f1,hepush_cost_effect,dqn_cost_effect)

    allcolumns = [[hepush_f1,dqn_f1,hepush_cost_effect,dqn_cost_effect]]

    df = pd.DataFrame(data=allcolumns,
                    columns=['Hepush-F1','DQN-F1','Hepush-CostEffect','DQN-CostEffect'])
    df.to_csv('./stastic_results_csv/dqtraffic_schedule_vs_dqn_performance.csv')

def calculate_delete_vs_accprofile_acc_with_time_slides(slide_nums=24):
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    hepush_upload_info = np.load(hepush_upload_path)
    dqn_upload_info = np.load(dqn_upload_path)

    hepush_delete_upload_info = np.load(hepush_delete_upload_path)
    hepush_accprofile_upload_info =  np.load(hepush_accprofile_upload_path)

    gt = small_pred!=large_pred
    hepush_delete_upload_info = np.append(hepush_delete_upload_info,1)
    hepush_accprofile_upload_info = np.append(hepush_accprofile_upload_info,0)
    acc1 = sum(gt==hepush_accprofile_upload_info)/len(gt)
    acc2 = sum(gt==hepush_delete_upload_info)/len(gt)
    print(acc1,acc2)


    gt_segs = np.array_split(gt,slide_nums)
    hepush_accprofile_segs = np.array_split(hepush_accprofile_upload_info,slide_nums)
    hepush_delete_segs = np.array_split(hepush_delete_upload_info,slide_nums)


    slides = [i for i in range(slide_nums)]
    allcolumns = []

    for i in slides:
        cur_accprofile_acc = sum(hepush_accprofile_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_delete_acc = sum(hepush_delete_segs[i]==gt_segs[i])/len(gt_segs[i])
        print(cur_accprofile_acc,cur_delete_acc)
        allcolumns.append([i,cur_accprofile_acc,cur_delete_acc])
    
    df = pd.DataFrame(data=allcolumns,
                    columns=['time_slides','hepush-accprofile','hepush-delete'])
    df.to_csv('./stastic_results_csv/dqtraffic_accprofile_vs_delete_accuracy_with_timeslides.csv')

    hepush_accprofile_sys_acc = sum([True if hepush_accprofile_upload_info[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)
    hepuesh_delete_sys_acc = sum([True if hepush_delete_upload_info[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)

    hepush_accprofile_uprate = sum(hepush_accprofile_upload_info)/len(small_pred)
    hepush_delete_uprate = sum(hepush_delete_upload_info)/len(small_pred)

    init_acc = 0.5613
    hepush_accprofile_cost_effect = (hepush_accprofile_sys_acc-init_acc)/hepush_accprofile_uprate
    hepush_delete_cost_effect = (hepuesh_delete_sys_acc-init_acc)/hepush_delete_uprate

    hepush_accprofile_f1 = f1_score(gt,hepush_accprofile_upload_info)
    hepush_delete_f1 = f1_score(gt,hepush_delete_upload_info)

    print(hepush_accprofile_f1,hepush_delete_f1,hepush_accprofile_cost_effect,hepush_delete_cost_effect)

    allcolumns = [[hepush_accprofile_f1,hepush_delete_f1,hepush_accprofile_cost_effect,hepush_delete_cost_effect]]

    df = pd.DataFrame(data=allcolumns,
                    columns=['Hepush-AccProf-F1','Hepush-Delete-F1','Hepush-AccProf-CostEffect','Hepush-Delete-CostEffect'])
    df.to_csv('./stastic_results_csv/dqtraffic_accfile_vs_delete_performance.csv')

def calculate_up1_and_up2_acc_with_time_slides(slide_nums=24):
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    # 设置需要对比的两个upload
    up1 = np.load(hepush_upload_path)
    up2 = np.load(hepush_halfreward_upload_path)

    gt = small_pred!=large_pred
    # up1 = np.append(up1,1)
    up2 = np.append(up2,0)
    # print(len(up1),len(up2))
    acc1 = sum(gt==up1)/len(gt)
    acc2 = sum(gt==up2)/len(gt)
    print(acc1,acc2)


    gt_segs = np.array_split(gt,slide_nums)
    up1_segs = np.array_split(up1,slide_nums)
    up2_segs = np.array_split(up2,slide_nums)


    slides = [i for i in range(slide_nums)]
    allcolumns = []

    for i in slides:
        cur_up1_acc = sum(up1_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_up2_acc = sum(up2_segs[i]==gt_segs[i])/len(gt_segs[i])
        print(cur_up1_acc,cur_up2_acc)
        allcolumns.append([i,cur_up1_acc,cur_up2_acc])
    
    df = pd.DataFrame(data=allcolumns,
                    columns=['time_slides','hepush-acc+bonus','hepush-acc'])
    df.to_csv('./stastic_results_csv/dqtraffic_bonusreward_vs_no_filtering_accuracy_with_timeslides.csv')

    up1_sys_acc = sum([True if up1[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)
    up2_sys_acc = sum([True if up2[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)

    up1_uprate = sum(up1)/len(small_pred)
    up2_uprate = sum(up2)/len(small_pred)

    init_acc = 0.5613
    up1_cost_effect = (up1_sys_acc-init_acc)/up1_uprate
    up2_cost_effect = (up2_sys_acc-init_acc)/up2_uprate

    up1_f1 = f1_score(gt,up1)
    up2_f1 = f1_score(gt,up2)

    print(up1_f1,up2_f1,up1_cost_effect,up2_cost_effect)

    allcolumns = [[up1_f1,up2_f1,up1_cost_effect,up2_cost_effect]]

    df = pd.DataFrame(data=allcolumns,
                    columns=['Hepush-acc+bonus-F1','Hepush-acc-F1','Hepush-acc+bonush-CostEffect','Hepush-acc-CostEffect'])
    df.to_csv('./stastic_results_csv/dqtraffic_bonusreward_vs_no_performance.csv')

def calculate_all_filters_system_accuracy():
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    hepush_upload_info = np.load(hepush_upload_path)
    # rtec_upload_info = np.load(rtec_upload_path)

    init_upload_info = np.zeros(len(small_pred))

    input_upload_info = np.load(input_upload_path)
    output_upload_info = np.load(output_upload_path)
    know_upload_info = np.load(know_upload_path)

    gt = small_pred!=large_pred
    hepush_upload_info = np.append(hepush_upload_info,1)
    
    init_sys_acc,init_cost = calculate_single_Acc_and_Cost(small_pred,large_pred,init_upload_info)
    hepush_sys_acc,hepush_cost = calculate_single_Acc_and_Cost(small_pred,large_pred,hepush_upload_info)
    
    input_sys_acc,input_cost = calculate_single_Acc_and_Cost(small_pred,large_pred,input_upload_info)
    output_sys_acc,output_cost = calculate_single_Acc_and_Cost(small_pred,large_pred,output_upload_info)
    know_sys_acc,know_cost = calculate_single_Acc_and_Cost(small_pred,large_pred,know_upload_info)

    allcolumns = [[init_sys_acc,init_cost,61.9],
                    [input_sys_acc,input_cost,61.9+0.117],
                    [output_sys_acc,output_cost,61.9+0.026],
                    [know_sys_acc,know_cost,61.9+0.00387],
                    [hepush_sys_acc,hepush_cost,61.9+0.567],
                    [1.0,0.0,140.7]]

    df = pd.DataFrame(data=allcolumns,
                    columns=['System Accuracy','Filter Comm. Cost','Filter Comp. Cost'])
    df.to_csv('./stastic_results_csv/dqtraffic_filters_computation_and_communication_and_sysacc.csv')





if __name__=='__main__':
    # calculate_all()
    # draw_acc_and_bound()
    # draw_sys_acc_with_training_size()
    # calculate_filters_acc_with_time_slides()
    # calculate_schedule_vs_dqn_acc_with_time_slides()
    # calculate_delete_vs_accprofile_acc_with_time_slides()
    # calculate_up1_and_up2_acc_with_time_slides()
    calculate_all_filters_system_accuracy()